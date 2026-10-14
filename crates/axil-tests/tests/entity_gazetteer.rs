//! Auto-linking finds the names the database's gazetteer knows: built-in
//! tools and services, the crates and dependencies of the project a
//! database belongs to, and the names of its `_deps` rows.

use axil_core::{Axil, Direction, RecordId};
use axil_graph::AxilBuilderGraphExt;
use serde_json::{json, Value};

fn open_with_graph(path: &std::path::Path) -> Axil {
    Axil::open(path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap()
}

/// `(source_text, entity_type)` of every entity a memory mentions.
fn mentioned(db: &Axil, id: &RecordId) -> Vec<(String, Value)> {
    db.neighbors(id, Some("mentions"), Direction::Out)
        .unwrap()
        .into_iter()
        .map(|r| {
            (
                r.data["source_text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                r.data["entity_type"].clone(),
            )
        })
        .collect()
}

fn has(found: &[(String, Value)], source: &str, ty: &str) -> bool {
    found.iter().any(|(s, t)| s == source && t == ty)
}

/// A workspace with two crates sharing a name and a few dependencies.
fn write_project(root: &std::path::Path) {
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\n[workspace.dependencies]\nredb = \"3\"\n\
         regex = \"1\"\n",
    )
    .unwrap();
    for name in ["acme-core", "acme-cli"] {
        let dir = root.join("crates").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\n[dependencies]\ntantivy = \"0.22\"\n"),
        )
        .unwrap();
    }
}

#[test]
fn a_project_database_links_its_crates_dependencies_and_known_tools() {
    let dir = tempfile::tempdir().unwrap();
    write_project(dir.path());
    std::fs::create_dir(dir.path().join(".axil")).unwrap();
    let db = open_with_graph(&dir.path().join(".axil").join("memory.axil"));
    let id = db
        .insert(
            "decisions",
            json!({ "summary": "Acme keeps redb behind acme-core and tantivy for search; \
                                CI runs on GitHub. A regex parses the log." }),
        )
        .unwrap()
        .id;
    let found = mentioned(&db, &id);
    for (source, ty) in [
        ("Acme", "project"),
        ("redb", "project"),
        ("acme-core", "code"),
        ("tantivy", "project"),
        ("GitHub", "project"),
    ] {
        assert!(has(&found, source, ty), "{source} ({ty}): {found:?}");
    }
    // A dependency named like a common word is not an entity.
    assert!(!found.iter().any(|(s, _)| s == "regex"), "{found:?}");
}

#[test]
fn a_database_outside_a_project_reads_no_manifest_but_its_deps_rows() {
    let dir = tempfile::tempdir().unwrap();
    // A manifest beside a loose database is not read: the directory may be
    // a home directory, not a project.
    write_project(dir.path());
    let path = dir.path().join("memory.axil");
    {
        let db = open_with_graph(&path);
        db.insert(
            "_deps",
            json!({ "name": "wasmtime-wasi", "version": "1.0.0", "status": "active" }),
        )
        .unwrap();
        db.insert(
            "_deps",
            json!({ "name": "old-crate", "version": "0.1.0", "status": "removed" }),
        )
        .unwrap();
    }
    let db = open_with_graph(&path);
    let id = db
        .insert(
            "decisions",
            json!({ "summary": "redb sits under wasmtime-wasi, not old-crate, on macOS" }),
        )
        .unwrap()
        .id;
    let found = mentioned(&db, &id);
    assert!(has(&found, "wasmtime-wasi", "project"), "{found:?}");
    assert!(has(&found, "macOS", "project"), "{found:?}");
    for name in ["redb", "old-crate"] {
        assert!(!found.iter().any(|(s, _)| s == name), "{name}: {found:?}");
    }
}
