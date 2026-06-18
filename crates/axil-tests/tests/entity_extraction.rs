//! Integration tests for entity extraction.

use axil_core::entity::{extract_entities, EntityType};

// ── Basic extraction ─────────────────────────────────────────────────────

#[test]
fn extract_backtick_entities() {
    let text = "Fixed bug in `AuthModule` by updating `auth_config`";
    let entities = extract_entities(text);
    let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    assert!(
        names.contains(&"AuthModule"),
        "missing AuthModule: {names:?}"
    );
    assert!(
        names.contains(&"auth_config"),
        "missing auth_config: {names:?}"
    );
}

#[test]
fn extract_file_paths() {
    let text = "Modified /src/auth/login.rs and ./config/settings.toml";
    let entities = extract_entities(text);
    let files: Vec<&str> = entities
        .iter()
        .filter(|e| e.entity_type == EntityType::File)
        .map(|e| e.name.as_str())
        .collect();
    assert!(!files.is_empty(), "expected file entities, got none");
    assert!(
        files.iter().any(|f| f.contains("auth/login.rs")),
        "missing login.rs: {files:?}"
    );
}

#[test]
fn extract_camel_case() {
    let text = "The UserAccountService handles all registration flows";
    let entities = extract_entities(text);
    let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    // CamelCase gets normalized to snake_case by the extractor.
    assert!(
        names
            .iter()
            .any(|n| n.contains("user_account_service") || n.contains("UserAccountService")),
        "missing CamelCase entity: {names:?}"
    );
}

#[test]
fn extract_snake_case() {
    let text = "Called process_payment_request to finalize the order";
    let entities = extract_entities(text);
    let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("process_payment_request")),
        "missing snake_case entity: {names:?}"
    );
}

// ── Deduplication ────────────────────────────────────────────────────────

#[test]
fn deduplicates_same_entity() {
    let text = "`AuthModule` is the AuthModule class in AuthModule";
    let entities = extract_entities(text);
    let auth_count = entities
        .iter()
        .filter(|e| e.name.to_lowercase().contains("authmodule") || e.name.contains("AuthModule"))
        .count();
    assert_eq!(auth_count, 1, "expected 1 AuthModule, got {auth_count}");
}

#[test]
fn deduplicates_case_insensitive() {
    let text = "Updated `auth_module` and also referenced AUTH_MODULE";
    let entities = extract_entities(text);
    let matches: Vec<&str> = entities
        .iter()
        .filter(|e| e.name.to_lowercase().contains("auth_module"))
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(matches.len(), 1, "expected 1 match, got: {matches:?}");
}

// ── Edge cases ───────────────────────────────────────────────────────────

#[test]
fn empty_text_returns_empty() {
    let entities = extract_entities("");
    assert!(entities.is_empty());
}

#[test]
fn no_entities_in_plain_text() {
    let text = "the cat sat on the mat";
    let entities = extract_entities(text);
    assert!(entities.is_empty(), "unexpected entities: {entities:?}");
}

#[test]
fn mixed_entity_types() {
    let text = "Modified `/src/db.rs` and `QueryBuilder` to fix query_timeout_handler";
    let entities = extract_entities(text);
    assert!(
        entities.len() >= 2,
        "expected at least 2 entities, got {}",
        entities.len()
    );

    let has_file = entities.iter().any(|e| e.entity_type == EntityType::File);
    let has_code = entities
        .iter()
        .any(|e| e.entity_type == EntityType::Code || e.entity_type == EntityType::Reference);
    assert!(
        has_file && has_code,
        "expected both file and code types in: {entities:?}"
    );
}

#[test]
fn entity_has_source_text() {
    let text = "Check `MyService` for details";
    let entities = extract_entities(text);
    let svc = entities.iter().find(|e| e.name == "MyService");
    assert!(svc.is_some(), "missing MyService");
    assert!(
        !svc.unwrap().source_text.is_empty(),
        "source_text should not be empty"
    );
}
