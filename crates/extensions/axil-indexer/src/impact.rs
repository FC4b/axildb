//! Graph-powered impact analysis for Axil.
//!
//! Uses the graph edges created by the project indexer (`module →contains→ file`,
//! `module →depends_on→ module`) to trace what is affected by a change to a
//! given file or symbol.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use axil_core::{Axil, Direction, Record, RecordId};

use crate::indexer::{TABLE_FILES, TABLE_MODULES};

/// Result of dependency analysis: (direct_dependents, transitive_dependents, affected_modules).
type DependencyResult = (Vec<String>, Vec<String>, Vec<String>);

// ── Types ──────────────────────────────────────────────────────────────

/// Impact analysis report for a single target file or symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactReport {
    /// The file/symbol being analyzed.
    pub target: String,
    /// Files that directly import or use this target.
    pub direct_dependents: Vec<String>,
    /// Files reachable through a transitive dependency chain.
    pub transitive_dependents: Vec<String>,
    /// Module names affected by the change.
    pub affected_modules: Vec<String>,
    /// Risk level: `"low"`, `"medium"`, or `"high"`.
    pub risk: String,
    /// Suggested action, e.g. which test suites to run.
    pub suggestion: String,
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Find a file record whose `path` field contains `target_path` (substring match).
fn find_file_record<'a>(files: &'a [Record], target_path: &str) -> Option<&'a Record> {
    files.iter().find(|r| {
        r.data
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| p.contains(target_path))
            .unwrap_or(false)
    })
}

/// Find the module record that contains a given file (via `contains` edge
/// pointing TO the file).
fn find_parent_module<'a>(
    db: &Axil,
    modules: &'a [Record],
    file_id: &RecordId,
) -> axil_core::Result<Option<&'a Record>> {
    let edges = db.edges(file_id, Some("contains"), Direction::In)?;
    if let Some(edge) = edges.first() {
        return Ok(modules.iter().find(|m| m.id == edge.from));
    }
    Ok(None)
}

/// Collect module names from a set of record IDs.
fn module_names_from(modules: &[Record], module_ids: &HashSet<RecordId>) -> Vec<String> {
    modules
        .iter()
        .filter(|m| module_ids.contains(&m.id))
        .filter_map(|m| {
            m.data
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .collect()
}

/// Collect file paths contained by a set of module IDs.
fn files_in_modules(db: &Axil, module_ids: &HashSet<RecordId>) -> axil_core::Result<Vec<String>> {
    let mut paths = Vec::new();
    for mid in module_ids {
        let neighbors = db.neighbors(mid, Some("contains"), Direction::Out)?;
        for rec in neighbors {
            if let Some(p) = rec.data.get("path").and_then(|v| v.as_str()) {
                paths.push(p.to_string());
            }
        }
    }
    Ok(paths)
}

/// Walk the `depends_on` graph transitively up to `max_depth` in the given
/// direction, starting from `start_module`. Returns all reachable module IDs
/// (excluding the start).
fn walk_dependency_graph(
    db: &Axil,
    start_module: &RecordId,
    direction: Direction,
    max_depth: usize,
) -> axil_core::Result<HashSet<RecordId>> {
    let mut visited: HashSet<RecordId> = HashSet::new();
    let mut queue: VecDeque<(RecordId, usize)> = VecDeque::new();
    queue.push_back((start_module.clone(), 0));
    visited.insert(start_module.clone());

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let edges = db.edges(&current, Some("depends_on"), direction)?;
        for edge in &edges {
            let next = match direction {
                Direction::In => &edge.from,
                Direction::Out => &edge.to,
                Direction::Both => {
                    // For both, take the "other" end.
                    if edge.from == current {
                        &edge.to
                    } else {
                        &edge.from
                    }
                }
            };
            if visited.insert(next.clone()) {
                queue.push_back((next.clone(), depth + 1));
            }
        }
    }

    // Remove the start node — we only want dependents.
    visited.remove(start_module);
    Ok(visited)
}

/// Classify risk based on the number of total dependents.
fn assess_risk(count: usize) -> &'static str {
    if count > 10 {
        "high"
    } else if count >= 3 {
        "medium"
    } else {
        "low"
    }
}

/// Build a suggestion string from affected module names.
fn build_suggestion(affected: &[String]) -> String {
    if affected.is_empty() {
        return "No downstream dependents detected — change is isolated.".to_string();
    }
    let names: Vec<&str> = affected.iter().map(|s| s.as_str()).collect();
    format!("Run {} tests after change", names.join(" + "))
}

fn empty_report(target: &str, suggestion: &str) -> ImpactReport {
    ImpactReport {
        target: target.to_string(),
        direct_dependents: vec![],
        transitive_dependents: vec![],
        affected_modules: vec![],
        risk: "low".to_string(),
        suggestion: suggestion.to_string(),
    }
}

// ── Shared analysis ───────────────────────────────────────────────────

/// Core dependency analysis shared by `impact()` and `reverse_impact()`.
///
/// `direction` controls which way the dependency graph is walked:
/// - `Direction::In` → who depends on this? (downstream impact)
/// - `Direction::Out` → what does this depend on? (upstream dependencies)
fn analyze_dependencies(
    db: &Axil,
    target_path: &str,
    direction: Direction,
) -> axil_core::Result<Option<DependencyResult>> {
    let files = db.list(TABLE_FILES)?;
    let modules = db.list(TABLE_MODULES)?;

    let file_rec = match find_file_record(&files, target_path) {
        Some(r) => r,
        None => return Ok(None),
    };

    let parent_module = find_parent_module(db, &modules, &file_rec.id)?;

    // Direct: one hop in the given direction.
    let mut direct_module_ids: HashSet<RecordId> = HashSet::new();
    if let Some(pm) = parent_module {
        let edges = db.edges(&pm.id, Some("depends_on"), direction)?;
        for edge in &edges {
            let id = match direction {
                Direction::In => &edge.from,
                Direction::Out => &edge.to,
                Direction::Both => {
                    if edge.from == pm.id {
                        &edge.to
                    } else {
                        &edge.from
                    }
                }
            };
            direct_module_ids.insert(id.clone());
        }
    }

    // Transitive: walk up to depth 3, excluding direct.
    let mut transitive_module_ids: HashSet<RecordId> = HashSet::new();
    if let Some(pm) = parent_module {
        transitive_module_ids = walk_dependency_graph(db, &pm.id, direction, 3)?;
        for did in &direct_module_ids {
            transitive_module_ids.remove(did);
        }
    }

    let direct_files = files_in_modules(db, &direct_module_ids)?;
    let transitive_files = files_in_modules(db, &transitive_module_ids)?;

    let mut all_affected_ids = direct_module_ids;
    all_affected_ids.extend(transitive_module_ids);
    if let Some(pm) = parent_module {
        all_affected_ids.insert(pm.id.clone());
    }
    let affected = module_names_from(&modules, &all_affected_ids);

    Ok(Some((direct_files, transitive_files, affected)))
}

// ── Public API ─────────────────────────────────────────────────────────

/// Analyze the downstream impact of changing `target_path`.
pub fn impact(db: &Axil, target_path: &str) -> axil_core::Result<ImpactReport> {
    if !db.has_graph_index() {
        return Ok(empty_report(
            target_path,
            "Enable graph plugin for impact analysis",
        ));
    }

    match analyze_dependencies(db, target_path, Direction::In)? {
        None => Ok(empty_report(
            target_path,
            &format!("File '{}' not found in index", target_path),
        )),
        Some((direct_files, transitive_files, affected)) => {
            let total = direct_files.len() + transitive_files.len();
            Ok(ImpactReport {
                target: target_path.to_string(),
                direct_dependents: direct_files,
                transitive_dependents: transitive_files,
                risk: assess_risk(total).to_string(),
                suggestion: build_suggestion(&affected),
                affected_modules: affected,
            })
        }
    }
}

/// Analyze what `target_path` depends ON (inverse of `impact`).
pub fn reverse_impact(db: &Axil, target_path: &str) -> axil_core::Result<ImpactReport> {
    if !db.has_graph_index() {
        return Ok(empty_report(
            target_path,
            "Enable graph plugin for impact analysis",
        ));
    }

    match analyze_dependencies(db, target_path, Direction::Out)? {
        None => Ok(empty_report(
            target_path,
            &format!("File '{}' not found in index", target_path),
        )),
        Some((direct_files, transitive_files, affected)) => {
            let total = direct_files.len() + transitive_files.len();
            let suggestion = if affected.is_empty() {
                "No upstream dependencies — this file is self-contained.".to_string()
            } else {
                format!("Changes in {} may break this file", affected.join(" + "))
            };
            Ok(ImpactReport {
                target: target_path.to_string(),
                direct_dependents: direct_files,
                transitive_dependents: transitive_files,
                risk: assess_risk(total).to_string(),
                suggestion,
                affected_modules: affected,
            })
        }
    }
}

/// Find and explain the connection path between two files.
///
/// Performs a BFS through graph edges (up to 5 hops) and returns a
/// human-readable chain such as:
/// `"jwt.rs ->imported_by-> middleware.rs ->imported_by-> routes.rs"`.
///
/// Returns an empty vec if no connection is found.
pub fn why_connected(db: &Axil, path_a: &str, path_b: &str) -> axil_core::Result<Vec<String>> {
    if !db.has_graph_index() {
        return Ok(vec![]);
    }

    let all_files = db.list(TABLE_FILES)?;
    let all_modules = db.list(TABLE_MODULES)?;

    let file_a = match find_file_record(&all_files, path_a) {
        Some(r) => r,
        None => return Ok(vec![]),
    };
    let file_b = match find_file_record(&all_files, path_b) {
        Some(r) => r,
        None => return Ok(vec![]),
    };

    const MAX_DEPTH: usize = 5;

    let mut visited: HashSet<RecordId> = HashSet::new();
    visited.insert(file_a.id.clone());

    // Build an ID → display-name map for readable output.
    let mut names: HashMap<RecordId, String> = HashMap::new();
    for f in &all_files {
        let label = f.data.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let short = label.rsplit('/').next().unwrap_or(label);
        names.insert(f.id.clone(), short.to_string());
    }
    for m in &all_modules {
        let label = m.data.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        names.insert(m.id.clone(), label.to_string());
    }

    struct BfsEntry {
        id: RecordId,
        chain: Vec<String>,
    }

    let mut queue: VecDeque<BfsEntry> = VecDeque::new();
    queue.push_back(BfsEntry {
        id: file_a.id.clone(),
        chain: vec![],
    });

    while let Some(entry) = queue.pop_front() {
        if entry.chain.len() / 2 >= MAX_DEPTH {
            continue;
        }

        // Explore all edges in both directions.
        let edges = db.edges(&entry.id, None, Direction::Both)?;
        for edge in &edges {
            let (next_id, arrow) = if edge.from == entry.id {
                (&edge.to, format!(" ->{}-> ", edge.edge_type))
            } else {
                (&edge.from, format!(" <-{}<- ", edge.edge_type))
            };

            if visited.contains(next_id) {
                continue;
            }
            visited.insert(next_id.clone());

            let current_name = names
                .get(&entry.id)
                .cloned()
                .unwrap_or_else(|| entry.id.0.clone());
            let next_name = names
                .get(next_id)
                .cloned()
                .unwrap_or_else(|| next_id.0.clone());

            let mut new_chain = entry.chain.clone();
            if new_chain.is_empty() {
                new_chain.push(current_name);
            }
            new_chain.push(arrow);
            new_chain.push(next_name.clone());

            if *next_id == file_b.id {
                // Found a path — collapse into a single readable string.
                return Ok(vec![new_chain.concat()]);
            }

            queue.push_back(BfsEntry {
                id: next_id.clone(),
                chain: new_chain,
            });
        }
    }

    Ok(vec![])
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_low_when_few_dependents() {
        assert_eq!(assess_risk(0), "low");
        assert_eq!(assess_risk(1), "low");
        assert_eq!(assess_risk(2), "low");
    }

    #[test]
    fn risk_medium_for_moderate_dependents() {
        assert_eq!(assess_risk(3), "medium");
        assert_eq!(assess_risk(5), "medium");
        assert_eq!(assess_risk(10), "medium");
    }

    #[test]
    fn risk_high_for_many_dependents() {
        assert_eq!(assess_risk(11), "high");
        assert_eq!(assess_risk(50), "high");
        assert_eq!(assess_risk(100), "high");
    }

    #[test]
    fn suggestion_empty_modules() {
        let s = build_suggestion(&[]);
        assert!(s.contains("isolated"));
    }

    #[test]
    fn suggestion_lists_modules() {
        let s = build_suggestion(&["auth".to_string(), "api".to_string()]);
        assert!(s.contains("auth"));
        assert!(s.contains("api"));
        assert!(s.contains("tests"));
    }

    #[test]
    fn empty_report_defaults() {
        let r = empty_report("foo.rs", "test");
        assert_eq!(r.target, "foo.rs");
        assert_eq!(r.risk, "low");
        assert!(r.direct_dependents.is_empty());
        assert!(r.transitive_dependents.is_empty());
        assert!(r.affected_modules.is_empty());
    }
}
