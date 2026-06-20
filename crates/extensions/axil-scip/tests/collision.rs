//! Cross-language collision test — acceptance.
//!
//! `login` defined in both a Rust document and a Python document must
//! produce two distinct `_entities` rows with distinct canonical ids.
//! Auto-merge would be a silent data-integrity regression.

use axil_core::Axil;
use axil_graph::AxilBuilderGraphExt;
use axil_scip::{
    ingest_scip,
    proto::{self, symbol_role},
};
use prost::Message;

fn rust_py_index() -> proto::Index {
    let rust_login = "rust-analyzer cargo app 0.1 auth/login().";
    let py_login = "scip-python python app 1.0 auth/login().";

    let rust_doc = proto::Document {
        language: "Rust".into(),
        relative_path: "src/auth.rs".into(),
        occurrences: vec![proto::Occurrence {
            range: vec![1, 0, 10],
            symbol: rust_login.into(),
            symbol_roles: symbol_role::DEFINITION,
            enclosing_range: vec![1, 0, 5, 0],
            ..Default::default()
        }],
        symbols: vec![proto::SymbolInformation {
            symbol: rust_login.into(),
            display_name: "login".into(),
            kind: 17,
            ..Default::default()
        }],
        ..Default::default()
    };
    let py_doc = proto::Document {
        language: "Python".into(),
        relative_path: "app/auth.py".into(),
        occurrences: vec![proto::Occurrence {
            range: vec![1, 0, 10],
            symbol: py_login.into(),
            symbol_roles: symbol_role::DEFINITION,
            enclosing_range: vec![1, 0, 5, 0],
            ..Default::default()
        }],
        symbols: vec![proto::SymbolInformation {
            symbol: py_login.into(),
            display_name: "login".into(),
            kind: 17,
            ..Default::default()
        }],
        ..Default::default()
    };

    proto::Index {
        metadata: Some(proto::Metadata {
            version: 0,
            tool_info: Some(proto::ToolInfo {
                name: "mixed".into(),
                version: "1".into(),
                arguments: vec![],
            }),
            project_root: "file:///tmp/mix".into(),
            text_document_encoding: 1,
        }),
        documents: vec![rust_doc, py_doc],
        external_symbols: vec![],
    }
}

#[test]
fn cross_language_login_does_not_merge() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("t.axil");
    let db = Axil::open(&db_path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();

    let index = rust_py_index();
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let scip_path = dir.path().join("index.scip");
    std::fs::write(&scip_path, &buf).unwrap();

    let report = ingest_scip(&db, &scip_path).unwrap();
    assert!(report.applied);

    let rows = db.list("_entities").unwrap();
    let logins: Vec<_> = rows
        .iter()
        .filter(|r| r.data.get("name").and_then(|v| v.as_str()) == Some("login"))
        .collect();
    assert_eq!(
        logins.len(),
        2,
        "rust + python `login` must stay distinct; got {:?}",
        logins
    );
    let canonical_ids: std::collections::HashSet<&str> = logins
        .iter()
        .filter_map(|r| r.data.get("canonical_id").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(
        canonical_ids.len(),
        2,
        "both rows must have distinct canonical_ids"
    );
}
