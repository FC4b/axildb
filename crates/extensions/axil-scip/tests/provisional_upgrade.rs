//! ↔ 13.2 checkpoint: provisional entities written by the regex
//! extractor must be rewritten to SCIP-grounded canonical ids when ingest
//! provides an unambiguous match.

use axil_core::entity::provisional_canonical_id;
use axil_core::{extract_entities, Axil};
use axil_graph::AxilBuilderGraphExt;
use axil_scip::{
    ingest_scip,
    proto::{self, symbol_role},
};
use prost::Message;
use serde_json::json;

#[test]
fn provisional_is_rewritten_by_scip() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("t.axil");
    let db = Axil::open(&db_path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();

    // Step 1 — hand-insert a provisional entity as though the regex
    // extractor had just run on the text `def login(u): pass`.
    let entities = extract_entities("def login(u): pass");
    let login = entities
        .iter()
        .find(|e| e.name == "login")
        .expect("python `login` must be extracted");
    let lang_hint = match &login.entity_type {
        axil_core::EntityType::CodeSymbol { lang_hint } => lang_hint.as_deref(),
        _ => None,
    };
    let provisional_id = provisional_canonical_id(&login.name, lang_hint, Some("app/auth.py"));
    assert!(provisional_id.starts_with("provisional:"));
    let inserted = db
        .insert(
            "_entities",
            json!({
                "name": "login",
                "canonical_id": provisional_id,
                "entity_type": { "code_symbol": { "lang_hint": "python" } },
                "source": "regex",
            }),
        )
        .unwrap();

    // Step 2 — build a SCIP index where `login` is defined in Python.
    let py_login = "scip-python python app 1.0 auth/login().";
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
    let index = proto::Index {
        metadata: Some(proto::Metadata {
            version: 0,
            tool_info: Some(proto::ToolInfo {
                name: "scip-python".into(),
                version: "0.4.0".into(),
                arguments: vec![],
            }),
            project_root: "file:///tmp/app".into(),
            text_document_encoding: 1,
        }),
        documents: vec![py_doc],
        external_symbols: vec![],
    };
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let scip_path = dir.path().join("index.scip");
    std::fs::write(&scip_path, &buf).unwrap();

    // Step 3 — ingest.
    let report = ingest_scip(&db, &scip_path).unwrap();
    assert!(
        report.provisional_upgraded >= 1,
        "expected 1 upgrade, got report = {:?}",
        report
    );

    // Step 4 — the row that was provisional now carries the SCIP canonical id.
    let row = db.get(&inserted.id).unwrap().expect("row still exists");
    let canonical_now = row
        .data
        .get("canonical_id")
        .and_then(|v| v.as_str())
        .unwrap();
    assert_eq!(canonical_now, py_login);
    assert!(
        !canonical_now.starts_with("provisional:"),
        "row still provisional after ingest"
    );
}
