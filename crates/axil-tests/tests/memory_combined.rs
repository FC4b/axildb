//! Combined integration tests — full agent workflow using all memory types.

use axil_core::Axil;
use axil_memory::preference::PreferenceSource;
use axil_memory::{AgentMemory, Outcome};
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

/// Simulates a complete agent session workflow using all memory types.
#[test]
fn full_agent_workflow() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    // 1. Set up rules (preference memory).
    mem.preference()
        .set(
            "error_handling",
            "Use thiserror in libs, anyhow in bins",
            PreferenceSource::User,
        )
        .unwrap();
    mem.preference()
        .set(
            "test_style",
            "Integration tests in tests/, unit tests in-file",
            PreferenceSource::User,
        )
        .unwrap();

    // 2. Store known facts (semantic memory).
    mem.semantic()
        .know(
            "auth-module",
            "Uses JWT tokens with 1h expiry, refresh token rotation enabled",
            None,
        )
        .unwrap();
    mem.semantic()
        .know("user-table", "PostgreSQL, columns: id, email, role", None)
        .unwrap();

    // 3. Start a session (working memory).
    let session = mem
        .working()
        .start_session(Some(json!({"task": "fix auth timeout"})))
        .unwrap();

    // 4. Log turns.
    mem.working()
        .log_turn(&session.id, "user", "Fix the auth timeout bug")
        .unwrap();
    mem.working()
        .log_turn(
            &session.id,
            "assistant",
            "I'll check the connection pool configuration",
        )
        .unwrap();
    mem.working()
        .log_turn(
            &session.id,
            "user",
            "Also check if the pool size is adequate for our load",
        )
        .unwrap();

    // 5. Log a decision (working memory).
    mem.working()
        .log(
            &session.id,
            "_working",
            json!({"decision": "Increase pool from 5 to 20"}),
        )
        .unwrap();

    // 6. End session → creates episode.
    let result = mem
        .working()
        .end_session(
            &session.id,
            Some("Fixed auth timeout. Root cause: pool exhaustion under load"),
            Some(Outcome::Success),
            Some(vec![
                "Increased pool from 5 to 20".into(),
                "Added connection timeout".into(),
            ]),
            Some(vec!["config.rs".into(), "pool.rs".into()]),
        )
        .unwrap();

    assert!(result.episode.is_some());
    let episode = result.episode.unwrap();

    // 7. Auto-extract procedure from episode.
    let procedure = mem.procedural().extract_from_episode(&episode).unwrap();
    assert!(procedure.is_some());

    // 8. Store updated knowledge (semantic memory).
    mem.semantic()
        .know(
            "auth-module",
            "Connection pool increased to 20 for production load",
            None,
        )
        .unwrap();

    // 9. Verify: check all memory types have content.
    let rules = mem.preference().list().unwrap();
    assert_eq!(rules.len(), 2);

    let entities = mem.semantic().list_entities().unwrap();
    assert!(entities.contains(&"auth-module".to_string()));
    assert!(entities.contains(&"user-table".to_string()));

    let episodes = mem.episodic().list(None, 100).unwrap();
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].data["outcome"], "success");

    let procedures = mem.procedural().list().unwrap();
    assert!(procedures.len() >= 1);

    let sessions = mem.working().list_sessions(false).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].data["status"], "ended");

    // 10. Verify: entity history shows evolution.
    let auth_history = mem.semantic().history("auth-module").unwrap();
    assert!(auth_history.len() >= 2);
}

/// Tests that multiple sessions accumulate episodes.
#[test]
fn multiple_sessions_accumulate() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    for i in 0..3 {
        let session = mem.working().start_session(None).unwrap();
        mem.working()
            .end_session(
                &session.id,
                Some(&format!("Session {i} summary")),
                Some(Outcome::Success),
                None,
                None,
            )
            .unwrap();
    }

    let episodes = mem.episodic().list(None, 100).unwrap();
    assert_eq!(episodes.len(), 3);

    let sessions = mem.working().list_sessions(false).unwrap();
    assert_eq!(sessions.len(), 3);
}

/// Tests preference override precedence.
#[test]
fn preference_override_precedence() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    // Detected preference.
    mem.preference()
        .set("indent", "tabs", PreferenceSource::Detected)
        .unwrap();

    // User preference overrides.
    mem.preference()
        .set("indent", "2 spaces", PreferenceSource::User)
        .unwrap();

    let rule = mem.preference().get("indent").unwrap().unwrap();
    assert_eq!(rule.data["value"], "2 spaces");

    // Detected should NOT override user.
    mem.preference()
        .set("indent", "4 spaces", PreferenceSource::Detected)
        .unwrap();

    let rule = mem.preference().get("indent").unwrap().unwrap();
    assert_eq!(rule.data["value"], "2 spaces");
}

/// Tests procedural memory confidence over multiple outcomes.
#[test]
fn procedural_confidence_evolution() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let proc = mem
        .procedural()
        .learn("test-approach", "Try A, then B, then C", None)
        .unwrap();

    let initial_conf = proc
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();

    // Three successes should boost confidence significantly.
    for _ in 0..3 {
        mem.procedural()
            .record_outcome(&proc.id, Outcome::Success)
            .unwrap();
    }

    let after_successes = mem
        .procedural()
        .find_by_name("test-approach")
        .unwrap()
        .unwrap();
    let success_conf = after_successes
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(success_conf > initial_conf + 0.2);

    // One failure should decrease but not zero out.
    mem.procedural()
        .record_outcome(&proc.id, Outcome::Failure)
        .unwrap();

    let after_failure = mem
        .procedural()
        .find_by_name("test-approach")
        .unwrap()
        .unwrap();
    let fail_conf = after_failure
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(fail_conf < success_conf);
    assert!(fail_conf > 0.0);
}
