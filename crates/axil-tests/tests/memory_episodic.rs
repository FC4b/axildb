//! Integration tests for episodic memory.

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
fn create_episode_directly() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let episode = mem
        .episodic()
        .create(
            "Fixed auth timeout by increasing connection pool",
            Outcome::Success,
            Some(vec!["Increased pool from 5 to 20".into()]),
            Some(vec!["config.rs".into()]),
        )
        .unwrap();

    assert_eq!(
        episode.data["summary"],
        "Fixed auth timeout by increasing connection pool"
    );
    assert_eq!(episode.data["outcome"], "success");
}

#[test]
fn list_episodes_by_outcome() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.episodic()
        .create("Success story", Outcome::Success, None, None)
        .unwrap();
    mem.episodic()
        .create("Failure story", Outcome::Failure, None, None)
        .unwrap();
    mem.episodic()
        .create("Partial story", Outcome::Partial, None, None)
        .unwrap();

    let successes = mem.episodic().list(Some(Outcome::Success), 100).unwrap();
    assert_eq!(successes.len(), 1);
    assert_eq!(successes[0].data["outcome"], "success");

    let all = mem.episodic().list(None, 100).unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn session_to_episode_lifecycle() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    // Start session → log turns → end → episode created.
    let session = mem.working().start_session(None).unwrap();
    mem.working()
        .log_turn(&session.id, "user", "Fix the auth timeout")
        .unwrap();
    mem.working()
        .log_turn(&session.id, "assistant", "Checking config...")
        .unwrap();
    mem.working()
        .log_turn(&session.id, "user", "Also check pool size")
        .unwrap();

    let result = mem
        .working()
        .end_session(
            &session.id,
            Some("Fixed timeout by increasing pool"),
            Some(Outcome::Success),
            Some(vec!["Increased pool".into()]),
            None,
        )
        .unwrap();

    // Episode should exist with full_text from user turns.
    let episode = result.episode.unwrap();
    assert_eq!(episode.data["outcome"], "success");

    // Full text should contain user turns only.
    let full_text = episode.data["full_text"].as_str().unwrap();
    assert!(full_text.contains("Fix the auth timeout"));
    assert!(full_text.contains("Also check pool size"));
    assert!(!full_text.contains("Checking config...")); // No assistant turns.

    // Episode should be findable via list.
    let episodes = mem.episodic().list(None, 100).unwrap();
    assert_eq!(episodes.len(), 1);
}
