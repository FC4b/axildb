//! `axil ingest-scip` must register scoped aliases so downstream
//! `entity-resolve-scoped` / `recall-for-entity` can resolve display
//! names back to SCIP canonical ids. Without this, ingest produces a
//! graph nobody can query by display name.

use axil_core::Axil;
use axil_graph::AxilBuilderGraphExt;
use axil_scip::{
    ingest_scip,
    proto::{self, symbol_role},
};
use prost::Message;

fn fixture() -> (tempfile::TempDir, std::path::PathBuf, Axil) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("t.axil");
    let db = Axil::open(&db_path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();

    let sym = "rust-analyzer cargo myapp 0.1 auth/login().";
    let doc = proto::Document {
        language: "Rust".into(),
        relative_path: "src/auth.rs".into(),
        occurrences: vec![proto::Occurrence {
            range: vec![1, 0, 10],
            symbol: sym.into(),
            symbol_roles: symbol_role::DEFINITION,
            enclosing_range: vec![1, 0, 5, 0],
            ..Default::default()
        }],
        symbols: vec![proto::SymbolInformation {
            symbol: sym.into(),
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
                name: "scip-rust".into(),
                version: "0.1".into(),
                arguments: vec![],
            }),
            project_root: "file:///t".into(),
            text_document_encoding: 1,
        }),
        documents: vec![doc],
        external_symbols: vec![],
    };
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let scip_path = dir.path().join("index.scip");
    std::fs::write(&scip_path, &buf).unwrap();
    (dir, scip_path, db)
}

#[test]
fn ingest_registers_scoped_aliases() {
    let (_dir, scip_path, db) = fixture();
    ingest_scip(&db, &scip_path).unwrap();

    let sym = "rust-analyzer cargo myapp 0.1 auth/login().";
    let lang_hit = db.resolve_entity_alias("login", &["lang:rust"]).unwrap();
    assert_eq!(lang_hit.as_deref(), Some(sym), "lang scope must resolve");

    let file_hit = db
        .resolve_entity_alias("login", &["file:src/auth.rs"])
        .unwrap();
    assert_eq!(file_hit.as_deref(), Some(sym), "file scope must resolve");

    let global_hit = db.resolve_entity_alias("login", &["global"]).unwrap();
    assert_eq!(
        global_hit.as_deref(),
        Some(sym),
        "global scope must resolve"
    );
}

#[test]
fn alias_registration_is_idempotent() {
    let (_dir, scip_path, db) = fixture();
    ingest_scip(&db, &scip_path).unwrap();
    let aliases_before = db.list(axil_core::SCIP_ALIAS_TABLE).unwrap().len();
    ingest_scip(&db, &scip_path).unwrap();
    let aliases_after = db.list(axil_core::SCIP_ALIAS_TABLE).unwrap().len();
    assert_eq!(
        aliases_before, aliases_after,
        "re-ingest must not duplicate alias rows"
    );
}
