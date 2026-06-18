//! Integration tests for working memory (session lifecycle).

use axil_core::Axil;
use axil_memory::{AgentMemory, Outcome};
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

#[test]
fn session_start_creates_active_record() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let session = mem
        .working()
        .start_session(Some(json!({"task": "fix auth timeout"})))
        .unwrap();

    assert_eq!(session.data["status"], "active");
    assert_eq!(session.data["meta"]["task"], "fix auth timeout");
    assert_eq!(session.data["record_count"], 0);
}

#[test]
fn session_log_increments_count() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let session = mem.working().start_session(None).unwrap();

    mem.working()
        .log(
            &session.id,
            "_working",
            json!({"tool": "grep", "result": "found"}),
        )
        .unwrap();
    mem.working()
        .log(
            &session.id,
            "_working",
            json!({"tool": "read", "file": "config.rs"}),
        )
        .unwrap();

    let updated = db.get(&session.id).unwrap().unwrap();
    assert_eq!(updated.data["record_count"], 2);
}

#[test]
fn session_turn_logging() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let session = mem.working().start_session(None).unwrap();
    mem.working()
        .log_turn(&session.id, "user", "Fix the auth timeout")
        .unwrap();
    mem.working()
        .log_turn(&session.id, "assistant", "Checking pool config...")
        .unwrap();
    mem.working()
        .log_turn(&session.id, "user", "Also check network settings")
        .unwrap();

    let updated = db.get(&session.id).unwrap().unwrap();
    let turns = updated.data["turns"].as_array().unwrap();
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[0]["role"], "user");
    assert_eq!(turns[2]["content"], "Also check network settings");
}

#[test]
fn session_end_creates_episode() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let session = mem.working().start_session(None).unwrap();
    mem.working()
        .log_turn(&session.id, "user", "Fix auth timeout")
        .unwrap();

    let result = mem
        .working()
        .end_session(
            &session.id,
            Some("Fixed auth timeout by increasing pool"),
            Some(Outcome::Success),
            Some(vec!["Increased pool from 5 to 20".into()]),
            Some(vec!["config.rs".into()]),
        )
        .unwrap();

    // Session should be ended.
    assert_eq!(result.session.data["status"], "ended");
    assert!(result.session.data.get("ended_at").is_some());
    assert!(result.session.data.get("duration_secs").is_some());

    // Episode should be created.
    assert!(result.episode.is_some());
    let episode = result.episode.unwrap();
    assert_eq!(episode.data["outcome"], "success");
    assert_eq!(
        episode.data["summary"],
        "Fixed auth timeout by increasing pool"
    );
}

#[test]
fn session_end_without_summary_no_episode() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let session = mem.working().start_session(None).unwrap();
    let result = mem
        .working()
        .end_session(&session.id, None, None, None, None)
        .unwrap();

    assert_eq!(result.session.data["status"], "ended");
    assert!(result.episode.is_none());
}

#[test]
fn cannot_log_to_ended_session() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let session = mem.working().start_session(None).unwrap();
    mem.working()
        .end_session(&session.id, None, None, None, None)
        .unwrap();

    let result = mem
        .working()
        .log(&session.id, "_working", json!({"test": true}));
    assert!(result.is_err());
}

#[test]
fn list_sessions_filters_active() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let s1 = mem.working().start_session(None).unwrap();
    let _s2 = mem.working().start_session(None).unwrap();
    mem.working()
        .end_session(&s1.id, None, None, None, None)
        .unwrap();

    let all = mem.working().list_sessions(false).unwrap();
    assert_eq!(all.len(), 2);

    let active = mem.working().list_sessions(true).unwrap();
    assert_eq!(active.len(), 1);
}
