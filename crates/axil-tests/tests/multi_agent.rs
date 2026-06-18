//! Integration tests for multi-agent record tagging and filtering.

use axil_core::Axil;
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

// ── Agent tagging ────────────────────────────────────────────────────────

#[test]
fn record_tagged_with_agent() {
    let (db, _dir) = temp_db();

    let mut data = json!({"summary": "fixed auth bug"});
    data.as_object_mut()
        .unwrap()
        .insert("_agent".to_string(), json!("code-agent"));

    let record = db.insert("decisions", data).unwrap();
    let retrieved = db.get(&record.id).unwrap().unwrap();

    assert_eq!(
        retrieved.data.get("_agent").and_then(|v| v.as_str()),
        Some("code-agent")
    );
}

#[test]
fn records_from_different_agents() {
    let (db, _dir) = temp_db();

    let mut d1 = json!({"summary": "from code agent"});
    d1.as_object_mut()
        .unwrap()
        .insert("_agent".to_string(), json!("code-agent"));

    let mut d2 = json!({"summary": "from review agent"});
    d2.as_object_mut()
        .unwrap()
        .insert("_agent".to_string(), json!("review-agent"));

    db.insert("decisions", d1).unwrap();
    db.insert("decisions", d2).unwrap();

    let all = db.list("decisions").unwrap();
    assert_eq!(all.len(), 2);

    let agents: Vec<&str> = all
        .iter()
        .filter_map(|r| r.data.get("_agent").and_then(|v| v.as_str()))
        .collect();
    assert!(agents.contains(&"code-agent"));
    assert!(agents.contains(&"review-agent"));
}

// ── Agent filtering ──────────────────────────────────────────────────────

#[test]
fn filter_by_agent_name() {
    let (db, _dir) = temp_db();

    for (agent, summary) in [
        ("agent-a", "task 1"),
        ("agent-a", "task 2"),
        ("agent-b", "task 3"),
    ] {
        let mut data = json!({"summary": summary});
        data.as_object_mut()
            .unwrap()
            .insert("_agent".to_string(), json!(agent));
        db.insert("work", data).unwrap();
    }

    let all = db.list("work").unwrap();
    let agent_a: Vec<_> = all
        .iter()
        .filter(|r| r.data.get("_agent").and_then(|v| v.as_str()) == Some("agent-a"))
        .collect();
    let agent_b: Vec<_> = all
        .iter()
        .filter(|r| r.data.get("_agent").and_then(|v| v.as_str()) == Some("agent-b"))
        .collect();

    assert_eq!(agent_a.len(), 2);
    assert_eq!(agent_b.len(), 1);
}

#[test]
fn untagged_records_have_no_agent() {
    let (db, _dir) = temp_db();

    let record = db.insert("notes", json!({"text": "no agent"})).unwrap();
    let retrieved = db.get(&record.id).unwrap().unwrap();

    assert!(
        retrieved.data.get("_agent").is_none(),
        "untagged record should not have _agent field"
    );
}

// ── Query builder with agent field filter ────────────────────────────────

#[test]
fn query_builder_filters_by_agent() {
    let (db, _dir) = temp_db();

    for (agent, n) in [("cursor", 3), ("claude", 2)] {
        for i in 0..n {
            let mut data = json!({"note": format!("{agent}-{i}")});
            data.as_object_mut()
                .unwrap()
                .insert("_agent".to_string(), json!(agent));
            db.insert("logs", data).unwrap();
        }
    }

    let results = db
        .query()
        .table("logs")
        .where_field("_agent", axil_core::Op::Eq, json!("cursor"))
        .exec()
        .unwrap();

    assert_eq!(
        results.len(),
        3,
        "expected 3 cursor records, got {}",
        results.len()
    );
    for r in &results {
        assert_eq!(
            r.data.get("_agent").and_then(|v| v.as_str()),
            Some("cursor")
        );
    }
}
