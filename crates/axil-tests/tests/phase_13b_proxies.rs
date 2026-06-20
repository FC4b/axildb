//! Integration tests for Phase 13b: structural code proxies.
//!
//! These tests run the full project indexer against a tiny on-disk
//! fixture, then verify that:
//! - `_idx_code_proxies` records are created (file/symbol/section).
//! - `proxy_id` is stable across re-indexing unchanged files.
//! - Adding lines above a symbol does not duplicate its logical proxy.
//! - Incremental re-index deletes stale proxies for changed/deleted files.
//! - `code_refs` stored on a memory survive line movement (match by
//!   `proxy_id`/`canonical_id` before path/line).

use std::fs;

use serde_json::{json, Value};

use axil_core::Axil;
use axil_indexer::recall::{recall_with_related, related_memories_for_proxies, RecallResult};
use axil_indexer::{IndexConfig, ProjectIndexer, TABLE_CODE_PROXIES, TABLE_CODE_REFS_INDEX};

fn temp_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

fn write_rust_fixture(root: &std::path::Path) {
    fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    let src = root.join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("lib.rs"),
        r#"//! Demo recall scoring entry point.

/// Combine vector and FTS scores into a final ranked list.
pub fn recall(query: &str, top_k: usize) -> Vec<u32> {
    let _ = (query, top_k);
    Vec::new()
}

/// Helper that does nothing yet.
pub fn vector_search(_q: &str) -> Vec<u32> {
    Vec::new()
}
"#,
    )
    .unwrap();
}

fn write_markdown_fixture(root: &std::path::Path) {
    fs::write(
        root.join("README.md"),
        r#"# Fixture Project

Intro paragraph.

## Recall Scoring

Recall combines vector + FTS + graph signals.

## Other

Unrelated content.
"#,
    )
    .unwrap();
}

fn proxies_for_path(db: &Axil, path: &str) -> Vec<Value> {
    db.list(TABLE_CODE_PROXIES)
        .unwrap()
        .into_iter()
        .filter(|r| r.data.get("path").and_then(|v| v.as_str()) == Some(path))
        .map(|r| r.data)
        .collect()
}

#[test]
fn full_index_creates_file_and_symbol_proxies() {
    let (db, dir) = temp_db();
    let root = dir.path();
    write_rust_fixture(root);
    write_markdown_fixture(root);

    let cfg = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, cfg);
    let result = indexer.index_full(root).unwrap();

    assert!(result
        .tables_created
        .iter()
        .any(|t| t == TABLE_CODE_PROXIES));

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    assert!(!proxies.is_empty(), "expected proxies, got 0");

    // Should include at least a file proxy for src/lib.rs and one symbol
    // proxy per pub fn.
    let lib_proxies = proxies_for_path(&db, "src/lib.rs");
    let kinds: Vec<&str> = lib_proxies
        .iter()
        .filter_map(|p| p.get("kind").and_then(|v| v.as_str()))
        .collect();
    assert!(
        kinds.iter().any(|k| *k == "file"),
        "missing file proxy: {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| *k == "symbol"),
        "missing symbol proxy: {kinds:?}"
    );

    // Markdown sections.
    let md_proxies = proxies_for_path(&db, "README.md");
    assert!(
        md_proxies
            .iter()
            .any(|p| p.get("kind").and_then(|v| v.as_str()) == Some("section")),
        "expected at least one markdown section proxy"
    );
    let recall_section = md_proxies
        .iter()
        .find(|p| p.get("symbol").and_then(|v| v.as_str()) == Some("Recall Scoring"));
    assert!(
        recall_section.is_some(),
        "README 'Recall Scoring' section missing"
    );

    // proxy_id is present on every record
    for p in &proxies {
        assert!(
            p.data.get("proxy_id").is_some(),
            "proxy_id missing on {:?}",
            p.data
        );
    }
}

#[test]
fn proxy_id_stable_across_full_reindex_and_line_movement() {
    let (db, dir) = temp_db();
    let root = dir.path();
    write_rust_fixture(root);

    let cfg = IndexConfig::default();
    ProjectIndexer::new(&db, cfg.clone())
        .index_full(root)
        .unwrap();

    let before: std::collections::HashMap<String, String> = db
        .list(TABLE_CODE_PROXIES)
        .unwrap()
        .into_iter()
        .filter(|r| r.data.get("kind").and_then(|v| v.as_str()) == Some("symbol"))
        .filter_map(|r| {
            let sym = r.data.get("symbol").and_then(|v| v.as_str())?.to_string();
            let pid = r.data.get("proxy_id").and_then(|v| v.as_str())?.to_string();
            Some((sym, pid))
        })
        .collect();
    assert!(!before.is_empty());

    // Move every symbol down by 50 lines: insert a doc-comment block at
    // the top of lib.rs without changing the symbol or its signature.
    let lib_path = root.join("src/lib.rs");
    let original = fs::read_to_string(&lib_path).unwrap();
    let mut prefix = String::new();
    for _ in 0..50 {
        prefix.push_str("// extra noise line\n");
    }
    fs::write(&lib_path, format!("{prefix}{original}")).unwrap();

    // Re-index incrementally. proxy_ids should match because identity is
    // (project, path, kind, symbol/canonical, signature_hash) — line is
    // just navigation metadata.
    ProjectIndexer::new(&db, cfg)
        .index_incremental(root)
        .unwrap();
    let after: std::collections::HashMap<String, String> = db
        .list(TABLE_CODE_PROXIES)
        .unwrap()
        .into_iter()
        .filter(|r| r.data.get("kind").and_then(|v| v.as_str()) == Some("symbol"))
        .filter_map(|r| {
            let sym = r.data.get("symbol").and_then(|v| v.as_str())?.to_string();
            let pid = r.data.get("proxy_id").and_then(|v| v.as_str())?.to_string();
            Some((sym, pid))
        })
        .collect();
    for (sym, before_id) in &before {
        let after_id = after.get(sym).expect("symbol disappeared after re-index");
        assert_eq!(before_id, after_id, "proxy_id changed for {sym}");
    }
}

#[test]
fn incremental_reindex_deletes_proxies_for_changed_files() {
    let (db, dir) = temp_db();
    let root = dir.path();
    write_rust_fixture(root);
    let cfg = IndexConfig::default();
    ProjectIndexer::new(&db, cfg.clone())
        .index_full(root)
        .unwrap();
    let initial_count = db.list(TABLE_CODE_PROXIES).unwrap().len();
    assert!(initial_count > 0);

    // Replace lib.rs with a single function — old symbols should disappear.
    fs::write(
        root.join("src/lib.rs"),
        r#"//! Replaced.

pub fn new_only() -> u32 { 0 }
"#,
    )
    .unwrap();
    ProjectIndexer::new(&db, cfg)
        .index_incremental(root)
        .unwrap();

    let after = db.list(TABLE_CODE_PROXIES).unwrap();
    let symbols: Vec<&str> = after
        .iter()
        .filter_map(|r| r.data.get("symbol").and_then(|v| v.as_str()))
        .collect();
    assert!(symbols.iter().any(|s| *s == "new_only"));
    assert!(!symbols.iter().any(|s| *s == "vector_search"));
    assert!(!symbols.iter().any(|s| *s == "recall"));
}

#[test]
fn pointer_attached_memories_match_by_proxy_id() {
    let (db, dir) = temp_db();
    let root = dir.path();
    write_rust_fixture(root);
    let cfg = IndexConfig::default();
    ProjectIndexer::new(&db, cfg).index_full(root).unwrap();

    // Pick the symbol proxy for `recall`.
    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let recall_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(|v| v.as_str()) == Some("recall"))
        .expect("recall proxy missing");
    let proxy_id = recall_proxy
        .data
        .get("proxy_id")
        .and_then(|v| v.as_str())
        .unwrap()
        .to_string();

    // Create a non-internal "decisions" table with a code_ref pointing at
    // the proxy. (`decisions` is just a normal user table.)
    db.insert(
        "decisions",
        json!({
            "summary": "Switched recall scoring to RRF",
            "code_refs": [
                {
                    "proxy_id": proxy_id,
                    "path": "src/lib.rs",
                    "symbol": "recall"
                }
            ]
        }),
    )
    .unwrap();

    // Build a fake proxy hit and ask for related memories.
    let hit = RecallResult {
        id: recall_proxy.id.to_string(),
        source: "proxy".to_string(),
        proxy_id: Some(proxy_id),
        path: Some("src/lib.rs".to_string()),
        symbol: Some("recall".to_string()),
        ..Default::default()
    };
    let related = related_memories_for_proxies(&db, &[hit], 5).unwrap();
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].source, "decisions");
    assert!(related[0].summary.contains("RRF"));
}

#[test]
fn scip_backfill_upgrades_canonical_id_and_proxy_id() {
    // Index a Rust fixture (no SCIP), then synthesize a `_scip_aliases`
    // row that maps `(src/lib.rs, recall) -> canonical_id`. Run the
    // backfill and assert:
    //   1. The proxy gains the canonical_id.
    //   2. proxy_id changes (identity now SCIP-grounded).
    //   3. Already-resolved proxies are left alone.
    //   4. Ambiguous matches (>1 candidate) are not upgraded.
    let (db, dir) = temp_db();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let recall_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(Value::as_str) == Some("recall"))
        .expect("recall proxy missing");
    let pre_proxy_id = recall_proxy.data["proxy_id"].as_str().unwrap().to_string();
    assert!(recall_proxy
        .data
        .get("canonical_id")
        .and_then(Value::as_str)
        .is_none());

    // Insert a SCIP alias matching this proxy.
    let canonical = "rust-analyzer cargo fixture 0.1.0 src/lib.rs/recall().";
    db.insert(
        axil_core::SCIP_ALIAS_TABLE,
        json!({
            "alias": "recall",
            "scope": "file:src/lib.rs",
            "canonical_id": canonical,
        }),
    )
    .unwrap();
    // Insert an ambiguous case for `vector_search`: two aliases sharing
    // path+name but distinct canonical_ids. Backfill should *not* pick
    // one — these should land in `ambiguous`.
    db.insert(
        axil_core::SCIP_ALIAS_TABLE,
        json!({"alias": "vector_search", "scope": "file:src/lib.rs", "canonical_id": "cid-A"}),
    )
    .unwrap();
    db.insert(
        axil_core::SCIP_ALIAS_TABLE,
        json!({"alias": "vector_search", "scope": "file:src/lib.rs", "canonical_id": "cid-B"}),
    )
    .unwrap();

    let report = axil_indexer::proxy::backfill_canonical_ids_from_scip(&db).unwrap();
    assert_eq!(report.upgraded, 1, "expected exactly recall to upgrade");
    assert_eq!(
        report.ambiguous, 1,
        "expected vector_search to be marked ambiguous"
    );

    // Re-fetch and verify.
    let after = db
        .list(TABLE_CODE_PROXIES)
        .unwrap()
        .into_iter()
        .find(|r| r.data.get("symbol").and_then(Value::as_str) == Some("recall"))
        .unwrap();
    assert_eq!(after.data["canonical_id"].as_str(), Some(canonical));
    let new_proxy_id = after.data["proxy_id"].as_str().unwrap();
    assert_ne!(
        new_proxy_id, pre_proxy_id,
        "proxy_id should change after canonical_id upgrade"
    );

    // Idempotent: running backfill again is a no-op for this proxy.
    let report2 = axil_indexer::proxy::backfill_canonical_ids_from_scip(&db).unwrap();
    assert_eq!(report2.upgraded, 0);
    // vector_search is still ambiguous and still unupgraded.
    assert_eq!(report2.ambiguous, 1);
}

#[test]
fn same_file_edges_emitted_when_graph_plugin_present() {
    use axil_core::{Axil, Direction};
    use axil_graph::AxilBuilderGraphExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let file_proxy = proxies
        .iter()
        .find(|r| {
            r.data.get("kind").and_then(Value::as_str) == Some("file")
                && r.data.get("path").and_then(Value::as_str) == Some("src/lib.rs")
        })
        .expect("file proxy missing");

    let same_file = db
        .edges(&file_proxy.id, Some("same_file"), Direction::Out)
        .unwrap();
    // The fixture has 2 pub fns: recall, vector_search.
    assert!(
        same_file.len() >= 2,
        "expected ≥2 same_file edges from file proxy, got {}",
        same_file.len()
    );

    // Recall expansion should surface symbol proxies from a file-proxy hit.
    let direct = vec![RecallResult {
        id: file_proxy.id.to_string(),
        source: "proxy".to_string(),
        path: Some("src/lib.rs".to_string()),
        symbol: None,
        kind: Some("file".to_string()),
        ..Default::default()
    }];
    let neighbors = axil_indexer::recall::graph_neighbors_for_proxies(&db, &direct, 5).unwrap();
    let symbols: Vec<&str> = neighbors
        .iter()
        .filter_map(|r| r.symbol.as_deref())
        .collect();
    assert!(
        symbols.contains(&"recall"),
        "neighbors missing recall: {symbols:?}"
    );
    assert!(
        symbols.contains(&"vector_search"),
        "neighbors missing vector_search: {symbols:?}"
    );
}

#[test]
fn tests_edges_emitted_for_test_files_by_naming_convention() {
    use axil_core::{Axil, Direction};
    use axil_graph::AxilBuilderGraphExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();

    // Single test file with both a target and its test. The `tests`
    // edge heuristic only resolves *intra-file* — cross-file SCIP
    // resolution is left to the canonical-id graph layer.
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"[package]
name = "fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    let tests = dir.path().join("tests");
    fs::create_dir_all(&tests).unwrap();
    fs::write(
        tests.join("login_test.rs"),
        r#"//! Login tests.
pub fn login(_u: &str) -> bool { true }
pub fn test_login() { assert!(login("a")); }
"#,
    )
    .unwrap();

    let mut cfg = IndexConfig::default();
    cfg.index_tests = true; // scanner skips tests by default
    ProjectIndexer::new(&db, cfg)
        .index_full(dir.path())
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let test_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(Value::as_str) == Some("test_login"))
        .expect("test_login proxy missing");
    let target_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(Value::as_str) == Some("login"))
        .expect("login target proxy missing");

    // `test_login` -> `login` via a `tests` edge (heuristic strips the
    // `test_` prefix and looks up a sibling proxy by name).
    let edges = db
        .edges(&test_proxy.id, Some("tests"), Direction::Out)
        .unwrap();
    assert!(
        edges.iter().any(|e| e.to == target_proxy.id),
        "expected `tests` edge from test_login to login; got edges: {:?}",
        edges
            .iter()
            .map(|e| (&e.edge_type, &e.to))
            .collect::<Vec<_>>()
    );
}

#[test]
fn graph_neighbors_expand_via_scip_edges() {
    // Build a DB with the graph plugin attached, an indexed Rust fixture,
    // and two `_entities` rows linked by a `calls` edge. Each entity has
    // a `canonical_id` that matches a proxy `canonical_id`. Then verify
    // that `graph_neighbors_for_proxies` surfaces the callee proxy when
    // the caller proxy is a direct hit.
    use axil_core::{Axil, Direction};
    use axil_graph::AxilBuilderGraphExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    // Tag the proxies with synthetic canonical ids so the entity bridge
    // has something to match.
    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let recall_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(Value::as_str) == Some("recall"))
        .unwrap();
    let vector_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(Value::as_str) == Some("vector_search"))
        .unwrap();
    let recall_cid = "rust fixture recall()";
    let vector_cid = "rust fixture vector_search()";
    {
        let mut data = recall_proxy.data.clone();
        data["canonical_id"] = Value::String(recall_cid.into());
        db.update(&recall_proxy.id, data).unwrap();
    }
    {
        let mut data = vector_proxy.data.clone();
        data["canonical_id"] = Value::String(vector_cid.into());
        db.update(&vector_proxy.id, data).unwrap();
    }

    let recall_entity = db
        .insert(
            "_entities",
            json!({"canonical_id": recall_cid, "name": "recall"}),
        )
        .unwrap();
    let vector_entity = db
        .insert(
            "_entities",
            json!({"canonical_id": vector_cid, "name": "vector_search"}),
        )
        .unwrap();
    db.relate(&recall_entity.id, "calls", &vector_entity.id, None)
        .unwrap();

    // Sanity: the edge is queryable.
    assert_eq!(
        db.edges(&recall_entity.id, Some("calls"), Direction::Out)
            .unwrap()
            .len(),
        1
    );

    // Direct hit: pretend `recall` matched. Verify `vector_search` arrives
    // as a graph neighbor.
    let direct = vec![RecallResult {
        id: recall_proxy.id.to_string(),
        source: "proxy".to_string(),
        path: Some("src/lib.rs".to_string()),
        symbol: Some("recall".to_string()),
        canonical_id: Some(recall_cid.to_string()),
        ..Default::default()
    }];
    let neighbors = axil_indexer::recall::graph_neighbors_for_proxies(&db, &direct, 5).unwrap();
    // With 13b.7 same_file edges enabled, the recall proxy now also has
    // a same_file neighbor (the file proxy and vector_search). Assert
    // that the SCIP `calls` neighbor is present and explains itself, and
    // that whatever else is present is also a legitimate neighbor.
    assert!(!neighbors.is_empty());
    let calls_neighbor = neighbors
        .iter()
        .find(|n| {
            n.symbol.as_deref() == Some("vector_search")
                && n.why.as_deref().unwrap_or("").contains("via calls")
        })
        .expect("missing vector_search neighbor via calls edge");
    assert!(calls_neighbor
        .why
        .as_deref()
        .unwrap_or("")
        .contains("graph neighbor"));
}

#[test]
fn toml_and_json_files_get_section_proxies() {
    let (db, dir) = temp_db();
    let root = dir.path();

    // Cargo-style TOML and a package.json-style JSON.
    fs::write(
        root.join("Cargo.toml"),
        r#"[package]
name = "fixture"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1"
tokio = "1"

[dev-dependencies]
tempfile = "3"
"#,
    )
    .unwrap();
    fs::write(
        root.join("package.json"),
        r#"{
  "name": "demo",
  "scripts": {
    "build": "tsc",
    "test": "jest"
  },
  "dependencies": {
    "react": "19"
  }
}"#,
    )
    .unwrap();

    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(root)
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let toml_sections: Vec<&str> = proxies
        .iter()
        .filter(|r| {
            r.data.get("path").and_then(Value::as_str) == Some("Cargo.toml")
                && r.data.get("kind").and_then(Value::as_str) == Some("section")
        })
        .filter_map(|r| r.data.get("symbol").and_then(Value::as_str))
        .collect();
    assert!(
        toml_sections.contains(&"package"),
        "missing TOML [package] section: {toml_sections:?}"
    );
    assert!(toml_sections.contains(&"dependencies"));
    assert!(toml_sections.contains(&"dev-dependencies"));

    let json_sections: Vec<&str> = proxies
        .iter()
        .filter(|r| {
            r.data.get("path").and_then(Value::as_str) == Some("package.json")
                && r.data.get("kind").and_then(Value::as_str) == Some("section")
        })
        .filter_map(|r| r.data.get("symbol").and_then(Value::as_str))
        .collect();
    assert!(
        json_sections.contains(&"scripts"),
        "missing JSON scripts section: {json_sections:?}"
    );
    assert!(json_sections.contains(&"dependencies"));
}

#[test]
fn yaml_ci_workflow_gets_section_proxies() {
    let (db, dir) = temp_db();
    let root = dir.path();
    // Use a non-hidden directory because the scanner walks with
    // `.hidden(true)` and skips dotfile dirs by default. Real CI files
    // live in `.github/workflows/`, which only get indexed when the
    // user sets the `include_hidden` flag — covered separately.
    let workflows = root.join("workflows");
    fs::create_dir_all(&workflows).unwrap();
    fs::write(
        workflows.join("ci.yml"),
        r#"name: CI

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: cargo build
"#,
    )
    .unwrap();

    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(root)
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let yaml_sections: Vec<&str> = proxies
        .iter()
        .filter(|r| {
            r.data
                .get("path")
                .and_then(Value::as_str)
                .map(|p| p.ends_with("ci.yml"))
                .unwrap_or(false)
                && r.data.get("kind").and_then(Value::as_str) == Some("section")
        })
        .filter_map(|r| r.data.get("symbol").and_then(Value::as_str))
        .collect();
    assert!(
        yaml_sections.contains(&"name"),
        "missing CI name: {yaml_sections:?}"
    );
    assert!(yaml_sections.contains(&"on"));
    assert!(yaml_sections.contains(&"jobs"));
    // The nested `build:` job must NOT spawn a top-level section.
    assert!(!yaml_sections.contains(&"build"));
}

#[test]
fn recall_with_related_returns_proxy_first() {
    let (db, dir) = temp_db();
    let root = dir.path();
    write_rust_fixture(root);
    write_markdown_fixture(root);
    let cfg = IndexConfig::default();
    ProjectIndexer::new(&db, cfg).index_full(root).unwrap();

    let rwr = recall_with_related(&db, "recall scoring", 5, 5).unwrap();
    // Without an embedder/FTS plugin the FTS-fused fast path returns nothing,
    // so we fall back to keyword matching which still surfaces the proxy.
    let any_proxy = rwr.primary.iter().any(|r| r.source == "proxy");
    assert!(
        any_proxy,
        "expected at least one proxy hit, got {:?}",
        rwr.primary
    );
}

#[test]
fn related_memories_use_reverse_index_when_present() {
    // `Axil::insert` auto-syncs `_idx_code_refs`, so a plain
    // db.insert("decisions", ...) is sufficient — no explicit indexer
    // call. related_memories_for_proxies should then resolve the memory
    // via the reverse index without walking every non-internal table.
    let (db, dir) = temp_db();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let recall_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(|v| v.as_str()) == Some("recall"))
        .expect("recall proxy missing");
    let proxy_id = recall_proxy
        .data
        .get("proxy_id")
        .and_then(|v| v.as_str())
        .unwrap()
        .to_string();

    db.insert(
        "decisions",
        json!({
            "summary": "Switched recall scoring to RRF",
            "code_refs": [{
                "proxy_id": proxy_id.clone(),
                "path": "src/lib.rs",
                "symbol": "recall",
            }],
        }),
    )
    .unwrap();

    let index_rows = db.list(TABLE_CODE_REFS_INDEX).unwrap();
    assert!(
        index_rows.iter().any(|r| {
            r.data.get("key").and_then(|v| v.as_str()) == Some(format!("proxy:{proxy_id}").as_str())
        }),
        "expected Axil::insert to auto-populate proxy: key in _idx_code_refs"
    );

    let hit = RecallResult {
        id: recall_proxy.id.to_string(),
        source: "proxy".to_string(),
        proxy_id: Some(proxy_id),
        path: Some("src/lib.rs".to_string()),
        symbol: Some("recall".to_string()),
        ..Default::default()
    };
    let related = related_memories_for_proxies(&db, &[hit], 5).unwrap();
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].source, "decisions");
    assert!(related[0].summary.contains("RRF"));
}

#[test]
fn updating_code_refs_replaces_stale_anchors() {
    // Update path: a memory whose code_refs change shouldn't accumulate
    // stale anchors. Axil::update drops prior rows and re-emits.
    let (db, dir) = temp_db();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    let inserted = db
        .insert(
            "decisions",
            json!({
                "summary": "first",
                "code_refs": [{"path": "src/lib.rs", "symbol": "recall"}],
            }),
        )
        .unwrap();
    let after_insert = db.list(TABLE_CODE_REFS_INDEX).unwrap();
    assert_eq!(after_insert.len(), 1);
    assert_eq!(
        after_insert[0].data.get("key").and_then(|v| v.as_str()),
        Some("path_symbol:src/lib.rs::recall")
    );

    db.update(
        &inserted.id,
        json!({
            "summary": "first",
            "code_refs": [{"path": "src/auth.rs"}],
        }),
    )
    .unwrap();
    let after_update = db.list(TABLE_CODE_REFS_INDEX).unwrap();
    assert_eq!(after_update.len(), 1);
    assert_eq!(
        after_update[0].data.get("key").and_then(|v| v.as_str()),
        Some("path:src/auth.rs")
    );
}

#[test]
fn mixed_db_old_memories_still_surface_via_fallback() {
    // Older memories written before the reverse-index hook landed have
    // `data.code_refs` but no `_idx_code_refs` rows. Once a NEW memory
    // populates the index, recall must still surface the old ones —
    // i.e. the indexed path must top up from the fallback walk when it
    // returns fewer than `limit` hits.
    let (db, dir) = temp_db();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap();
    let recall_proxy = proxies
        .iter()
        .find(|r| r.data.get("symbol").and_then(|v| v.as_str()) == Some("recall"))
        .expect("recall proxy missing");
    let proxy_id = recall_proxy
        .data
        .get("proxy_id")
        .and_then(|v| v.as_str())
        .unwrap()
        .to_string();

    // Simulate a pre-hook memory: insert directly via storage so the
    // auto-sync hook doesn't fire. The Axil instance can't bypass the
    // hook directly, so we use db.insert and then manually delete the
    // index rows it created — leaving `data.code_refs` intact on the
    // record but `_idx_code_refs` empty for that record_id.
    let old_memory = db
        .insert(
            "decisions",
            json!({
                "summary": "pre-hook decision about recall",
                "code_refs": [{"path": "src/lib.rs", "symbol": "recall"}],
            }),
        )
        .unwrap();
    let old_id_str = old_memory.id.to_string();
    for row in db.list(TABLE_CODE_REFS_INDEX).unwrap() {
        if row.data.get("record_id").and_then(|v| v.as_str()) == Some(old_id_str.as_str()) {
            db.delete(&row.id).unwrap();
        }
    }

    // New memory writes through the hook, populating the index.
    db.insert(
        "decisions",
        json!({
            "summary": "post-hook decision",
            "code_refs": [{"proxy_id": proxy_id.clone()}],
        }),
    )
    .unwrap();

    let hit = RecallResult {
        id: recall_proxy.id.to_string(),
        source: "proxy".to_string(),
        proxy_id: Some(proxy_id),
        path: Some("src/lib.rs".to_string()),
        symbol: Some("recall".to_string()),
        ..Default::default()
    };
    let related = related_memories_for_proxies(&db, &[hit], 5).unwrap();
    let summaries: Vec<&str> = related.iter().map(|r| r.summary.as_str()).collect();
    assert!(
        summaries.iter().any(|s| s.contains("pre-hook")),
        "expected pre-hook memory via fallback top-up, got {summaries:?}"
    );
    assert!(
        summaries.iter().any(|s| s.contains("post-hook")),
        "expected post-hook memory via index, got {summaries:?}"
    );
}

#[test]
fn deleting_a_memory_drops_its_anchor_rows() {
    // Delete path: removing a pointer-attached memory should evict its
    // reverse-index rows so recall doesn't surface tombstones.
    let (db, dir) = temp_db();
    write_rust_fixture(dir.path());
    ProjectIndexer::new(&db, IndexConfig::default())
        .index_full(dir.path())
        .unwrap();

    let inserted = db
        .insert(
            "decisions",
            json!({
                "summary": "doomed",
                "code_refs": [{"path": "src/lib.rs", "symbol": "recall"}],
            }),
        )
        .unwrap();
    assert_eq!(db.list(TABLE_CODE_REFS_INDEX).unwrap().len(), 1);

    db.delete(&inserted.id).unwrap();
    assert!(db.list(TABLE_CODE_REFS_INDEX).unwrap().is_empty());
}
