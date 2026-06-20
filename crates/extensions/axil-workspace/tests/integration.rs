//! Integration tests covering the cross-crate contract:
//! manifest + consent on `axil_core::Record` + bridges + blast radius.

use std::fs;

use axil_core::Axil;
use axil_workspace::bridge::{BridgeEvidence, EntityBridge};
use axil_workspace::consent::{MatchContext, ReadConsent, WriteConsent};
use axil_workspace::manifest::MANIFEST_FILENAME;
use axil_workspace::{discover_manifest, resolve_member};
use serde_json::json;
use tempfile::TempDir;

fn seed_workspace_with_dbs() -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    fs::create_dir_all(root.join("frontend/.axil")).unwrap();
    fs::create_dir_all(root.join("backend/.axil")).unwrap();

    let fe_path = root.join("frontend/.axil/memory.axil");
    let be_path = root.join("backend/.axil/memory.axil");
    let _ = Axil::open(&fe_path).build().unwrap();
    let _ = Axil::open(&be_path).build().unwrap();

    let manifest = format!(
        r#"
[workspace]
id = "ws_test"
name = "test"

[members.frontend]
id = "mem_fe"
root = "./frontend"
path = "./frontend/.axil/memory.axil"

[members.backend]
id = "mem_be"
root = "./backend"
path = "./backend/.axil/memory.axil"
"#
    );
    fs::write(root.join(MANIFEST_FILENAME), manifest).unwrap();
    (tmp, fe_path, be_path)
}

#[test]
fn manifest_discovery_and_member_resolve() {
    let (tmp, _fe, _be) = seed_workspace_with_dbs();
    let manifest = discover_manifest(tmp.path()).unwrap().unwrap();
    assert_eq!(manifest.members.len(), 2);

    let fe_dir = tmp.path().join("frontend");
    let res = resolve_member(&manifest, &fe_dir).unwrap();
    assert_eq!(res.member_label, "frontend");
    assert_eq!(res.member_id, "mem_fe");
}

#[test]
fn consent_defaults_respected_across_record_lifecycle() {
    let (_tmp, fe_path, _be) = seed_workspace_with_dbs();
    let db = Axil::open(&fe_path).build().unwrap();

    // Promote the `decisions` table to workspace-scoped.
    db.set_consent_default(
        "decisions",
        Some(serde_json::to_value(&ReadConsent::Workspace).unwrap()),
        None,
    )
    .unwrap();

    // Insert a record — defaults should land on it.
    let rec = db
        .insert("decisions", json!({"summary": "pick rust"}))
        .unwrap();
    let rec = db.storage().get(&rec.id).unwrap().unwrap();
    let parsed: ReadConsent = serde_json::from_value(rec.read_consent_raw()).unwrap();
    assert_eq!(parsed, ReadConsent::Workspace);

    // `consent set --read private` on that row flips it back.
    db.set_record_consent(
        &rec.id,
        Some(serde_json::to_value(&ReadConsent::Private).unwrap()),
        None,
    )
    .unwrap();
    let rec = db.storage().get(&rec.id).unwrap().unwrap();
    let reparsed: ReadConsent = serde_json::from_value(rec.read_consent_raw()).unwrap();
    assert_eq!(reparsed, ReadConsent::Private);

    // The consent change must be reflected in the dedicated audit log so
    // Compliance export has a stable trail.
    let log = db.list("_consent_log").unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(
        log[0].data.get("record_id").and_then(|v| v.as_str()),
        Some(rec.id.as_str())
    );
}

#[test]
fn consent_match_context_rejects_cross_member_private_reads() {
    let ws = "ws_test".to_string();
    let fe = "mem_fe".to_string();
    let be = "mem_be".to_string();
    let roles: Vec<String> = vec![];

    let ctx = MatchContext {
        source_workspace: &ws,
        source_member: &fe,
        caller_workspace: &ws,
        caller_member: &be,
        caller_roles: &roles,
        strict: false,
    };
    assert!(!ReadConsent::Private.allows(&ctx));
    assert!(ReadConsent::Workspace.allows(&ctx));
}

#[test]
fn source_only_write_consent_rejected_at_remote() {
    let ws = "ws_test".to_string();
    let fe = "mem_fe".to_string();
    let be = "mem_be".to_string();
    let roles: Vec<String> = vec![];
    let ctx = MatchContext {
        source_workspace: &ws,
        source_member: &fe,
        caller_workspace: &ws,
        caller_member: &be,
        caller_roles: &roles,
        strict: false,
    };
    let err = WriteConsent::SourceOnly
        .check("rec_1", &ctx)
        .expect_err("source-only should reject remote write");
    assert!(format!("{err:#}").contains("source_only") || format!("{err:#}").contains("mem_fe"));
}

#[test]
fn entity_bridges_round_trip_and_verify() {
    let (_tmp, fe_path, _be) = seed_workspace_with_dbs();
    let db = Axil::open(&fe_path).build().unwrap();

    let bridge = EntityBridge::new_manual("fe::login", "ws_test", "mem_be", "be::login", 0.9);
    db.upsert_bridge(&serde_json::to_value(&bridge).unwrap())
        .unwrap();
    // Upsert on the same identity must not produce a duplicate row.
    let bridge2 = EntityBridge::new_manual("fe::login", "ws_test", "mem_be", "be::login", 1.0);
    db.upsert_bridge(&serde_json::to_value(&bridge2).unwrap())
        .unwrap();

    let all = db.list_bridges(None, None).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(
        all[0]
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .round(),
        1.0
    );

    // Verify marks it dangling because the local canonical doesn't exist
    // in `_entities` on this fresh DB.
    let (verified, dangling) = db.verify_bridges().unwrap();
    assert_eq!(verified, 0);
    assert_eq!(dangling, 1);
    let all = db.list_bridges(None, None).unwrap();
    assert_eq!(
        all[0].data.get("dangling").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn bridge_evidence_confidence_tiers() {
    assert_eq!(
        BridgeEvidence::ScipSymbol { symbol: "x".into() }.default_confidence(),
        1.0
    );
    assert!(
        BridgeEvidence::NameAndType {
            name: "login".into(),
            type_name: "fn".into()
        }
        .default_confidence()
            < 0.5,
        "name-and-type must stay weak until promoted"
    );
}
