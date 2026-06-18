//! Integration tests for preference memory.

use axil_core::Axil;
use axil_memory::preference::PreferenceSource;
use axil_memory::AgentMemory;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

#[test]
fn set_and_get_rule() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.preference()
        .set(
            "error_handling",
            "Use thiserror in libs, anyhow in bins",
            PreferenceSource::User,
        )
        .unwrap();

    let rule = mem.preference().get("error_handling").unwrap();
    assert!(rule.is_some());
    let rule = rule.unwrap();
    assert_eq!(rule.data["value"], "Use thiserror in libs, anyhow in bins");
    assert_eq!(rule.data["source"], "user");
}

#[test]
fn user_overrides_detected() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.preference()
        .set("style", "detected_val", PreferenceSource::Detected)
        .unwrap();
    mem.preference()
        .set("style", "user_val", PreferenceSource::User)
        .unwrap();

    let rule = mem.preference().get("style").unwrap().unwrap();
    assert_eq!(rule.data["value"], "user_val");
    assert_eq!(rule.data["source"], "user");
}

#[test]
fn detected_does_not_override_user() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.preference()
        .set("style", "user_val", PreferenceSource::User)
        .unwrap();
    mem.preference()
        .set("style", "detected_val", PreferenceSource::Detected)
        .unwrap();

    let rule = mem.preference().get("style").unwrap().unwrap();
    // User rule should NOT be overridden.
    assert_eq!(rule.data["value"], "user_val");
}

#[test]
fn list_rules() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.preference()
        .set("a", "1", PreferenceSource::User)
        .unwrap();
    mem.preference()
        .set("b", "2", PreferenceSource::User)
        .unwrap();
    mem.preference()
        .set("c", "3", PreferenceSource::Detected)
        .unwrap();

    let all = mem.preference().list().unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn delete_rule() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.preference()
        .set("temp", "value", PreferenceSource::User)
        .unwrap();

    assert!(mem.preference().delete("temp").unwrap());
    assert!(mem.preference().get("temp").unwrap().is_none());

    // Delete nonexistent returns false.
    assert!(!mem.preference().delete("nonexistent").unwrap());
}

#[test]
fn extract_preferences_from_text() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let text = r#"
# Project Conventions
Always run tests before committing
Never push directly to main
Use thiserror for error handling in libs
Avoid global mutable state
I prefer functional style
"#;

    let extracted = mem.preference().extract_from_text(text).unwrap();
    assert!(
        extracted.len() >= 4,
        "should extract at least 4 preferences"
    );
}

#[test]
fn synthetic_doc_in_record() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.preference()
        .set("error_handling", "Use thiserror", PreferenceSource::User)
        .unwrap();

    let rule = mem.preference().get("error_handling").unwrap().unwrap();
    let doc = rule.data["synthetic_doc"].as_str().unwrap();
    assert!(doc.contains("error_handling"));
    assert!(doc.contains("thiserror"));
}
