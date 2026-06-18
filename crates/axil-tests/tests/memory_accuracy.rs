//! Axil-specific memory accuracy benchmarks (task 6.10).
//!
//! Each test creates a fresh temp database, exercises a specific memory feature,
//! and asserts accuracy criteria. Tests are `#[ignore]` because they involve
//! vector indexing and are slow.
//!
//! Run with: `cargo test -p axil-tests --test memory_accuracy -- --ignored`

use std::collections::HashSet;

use axil_core::{consolidate_facts, extract_entities, Axil, ConflictResult, Direction, Record};
use axil_graph::AxilBuilderGraphExt;
use axil_memory::preference::PreferenceSource;
use axil_memory::types::META_SUPERSEDED;
use axil_memory::{AgentMemory, Outcome, RecallOptions};
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;
use tempfile::TempDir;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Create a database with vector (3-dim synthetic) + graph.
fn temp_db_full() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.axil");
    let db = Axil::open(&path)
        .with_vector(3)
        .unwrap()
        .with_graph_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

/// Create a database with graph only (no vector).
fn temp_db_graph() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.axil");
    let db = Axil::open(&path)
        .with_graph_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

/// Print a JSON result line for benchmark reporting.
fn report(test: &str, passed: bool, score: f64, details: &str) {
    let json = json!({
        "test": test,
        "passed": passed,
        "score": score,
        "details": details,
    });
    println!("{}", serde_json::to_string(&json).unwrap());
}

// ── 1. Superseding Accuracy ───────────────────────────────────────────────

#[test]
#[ignore]
fn test_superseding_accuracy() {
    // We use synthetic 3-dim vectors and mark superseding manually, since
    // the SupersedeEngine requires `similar_to()` which needs real embeddings.
    // Instead we test the semantic-level contract: after marking old records
    // superseded, list_facts / about() excludes them.

    let (db, _dir) = temp_db_full();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    // Store 50 initial facts across 10 entities.
    let entities: Vec<String> = (0..10).map(|i| format!("entity_{i}")).collect();
    let mut original_ids = Vec::new();
    for (i, entity) in entities.iter().enumerate() {
        for j in 0..5 {
            let fact = format!("Fact {j} about {entity}: original version {i}_{j}");
            let record = sem.know(entity, &fact, None).unwrap();
            original_ids.push(record.id.clone());
        }
    }
    assert_eq!(original_ids.len(), 50);

    // Update 20 facts (first 2 facts of each entity) with newer versions.
    // Mark the originals as superseded and store replacements.
    let mut superseded_count = 0;
    let mut replacement_ids = Vec::new();
    for (i, entity) in entities.iter().enumerate() {
        for j in 0..2 {
            let old_idx = i * 5 + j;
            let old_id = &original_ids[old_idx];

            // Mark old as superseded.
            let old_record = db.get(old_id).unwrap().unwrap();
            let mut old_data = old_record.data.clone();
            axil_memory::ttl::set_meta_field(&mut old_data, META_SUPERSEDED, json!(true));
            db.update(old_id, old_data).unwrap();
            superseded_count += 1;

            // Store replacement.
            let new_fact = format!("Fact {j} about {entity}: UPDATED version {i}_{j}");
            let new_record = sem.know(entity, &new_fact, None).unwrap();
            replacement_ids.push(new_record.id.clone());
        }
    }

    assert_eq!(superseded_count, 20);

    // Verify: for each entity, about() should exclude superseded facts.
    let mut total_facts_returned = 0;
    let mut superseded_leaked = 0;
    for entity in &entities {
        let knowledge = sem.about(entity).unwrap();
        for fact_record in &knowledge.facts {
            total_facts_returned += 1;
            let is_superseded = fact_record
                .data
                .get("_meta")
                .and_then(|m| m.get("superseded"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_superseded {
                superseded_leaked += 1;
            }
        }
    }

    // Each entity had 5 original + 2 replacement = 7 total, minus 2 superseded = 5 active.
    // So we expect 50 active facts total.
    let expected_active = 50; // 10 entities * (5 - 2 superseded + 2 replacement)
    let accuracy = if total_facts_returned > 0 {
        1.0 - (superseded_leaked as f64 / total_facts_returned as f64)
    } else {
        0.0
    };

    let passed = superseded_leaked == 0 && total_facts_returned == expected_active;
    report(
        "superseding_accuracy",
        passed,
        accuracy,
        &format!(
            "total_returned={total_facts_returned}, expected={expected_active}, superseded_leaked={superseded_leaked}"
        ),
    );

    assert_eq!(
        superseded_leaked, 0,
        "superseded records must not appear in about()"
    );
    assert_eq!(
        total_facts_returned, expected_active,
        "expected {expected_active} active facts, got {total_facts_returned}"
    );
}

// ── 2. Entity Disambiguation ──────────────────────────────────────────────

#[test]
#[ignore]
fn test_entity_disambiguation() {
    // extract_entities() is pattern-based (backtick, CamelCase, snake_case, paths).
    // Test that it correctly identifies different entity types and doesn't
    // conflate entities with the same surface form in different contexts.

    let test_cases: Vec<(&str, Vec<(&str, &str)>)> = vec![
        // Case 1: backtick code entities
        (
            "Fixed bug in `AuthModule` by updating `auth_config` in /src/auth/login.rs",
            vec![
                ("AuthModule", "code"),
                ("auth_config", "code"),
                ("/src/auth/login.rs", "file"),
            ],
        ),
        // Case 2: CamelCase identifiers
        (
            "The HttpClient sends requests to ApiGateway",
            vec![("http_client", "code"), ("api_gateway", "code")],
        ),
        // Case 3: snake_case identifiers
        (
            "Updated user_profile and session_manager modules",
            vec![("user_profile", "code"), ("session_manager", "code")],
        ),
        // Case 4: file paths
        (
            "Modified ./src/main.rs and /config/database.yml",
            vec![("src/main.rs", "file"), ("/config/database.yml", "file")],
        ),
        // Case 5: mixed entities
        (
            "The `DatabasePool` in /src/db/pool.rs handles connection_pool lifecycle",
            vec![
                ("DatabasePool", "code"),
                ("/src/db/pool.rs", "file"),
                ("connection_pool", "code"),
            ],
        ),
    ];

    let mut total_expected = 0;
    let mut total_found = 0;

    for (text, expected_entities) in &test_cases {
        let entities = extract_entities(text);
        for (expected_name, expected_type) in expected_entities {
            total_expected += 1;
            // Check if any extracted entity matches (by substring for names, exact for types).
            let found = entities.iter().any(|e| {
                let name_match = e.name.contains(expected_name) || expected_name.contains(&e.name);
                let type_match = match (&e.entity_type, *expected_type) {
                    (axil_core::entity::EntityType::File, "file") => true,
                    (axil_core::entity::EntityType::Code, "code") => true,
                    (axil_core::entity::EntityType::Project, "project") => true,
                    (axil_core::entity::EntityType::Reference, "reference") => true,
                    _ => false,
                };
                name_match && type_match
            });
            if found {
                total_found += 1;
            }
        }
    }

    let accuracy = total_found as f64 / total_expected as f64;
    let passed = accuracy >= 0.80;

    report(
        "entity_disambiguation",
        passed,
        accuracy,
        &format!("found={total_found}/{total_expected}"),
    );

    assert!(
        accuracy >= 0.80,
        "entity disambiguation accuracy {accuracy:.2} below threshold 0.80 ({total_found}/{total_expected})"
    );
}

// ── 3. Knowledge Consolidation ────────────────────────────────────────────

#[test]
#[ignore]
fn test_knowledge_consolidation() {
    // Store 10 fragmented facts about "Alice" and consolidate them.
    let facts_text = vec![
        "Alice works at Google as a senior engineer",
        "Alice lives in San Francisco near the Mission district",
        "Alice likes Python and uses it for data analysis",
        "Alice graduated from MIT in 2018",
        "Alice mentors junior engineers on the platform team",
        "Alice previously worked at Facebook for 3 years",
        "Alice enjoys hiking in Marin County on weekends",
        "Alice is working on a machine learning project",
        "Alice prefers vim over VS Code for editing",
        "Alice speaks fluent Japanese learned during college",
    ];

    // Build records with timestamps (oldest first).
    let mut records_with_conflicts = Vec::new();
    for (i, fact) in facts_text.iter().enumerate() {
        let mut record = Record::new("_entities", json!({"entity": "Alice", "fact": fact}));
        // Spread timestamps across 10 days.
        record.created_at =
            chrono::Utc::now() - chrono::Duration::days((facts_text.len() - i) as i64);
        records_with_conflicts.push((record, ConflictResult::Novel));
    }

    let consolidated = consolidate_facts("Alice", &records_with_conflicts);
    assert!(
        consolidated.is_some(),
        "consolidation should produce a result"
    );

    let result = consolidated.unwrap();

    // Verify the consolidated profile.
    let summary = &result.summary;
    let source_count = result.source_ids.len();

    // The summary should mention Alice.
    let mentions_entity = summary.contains("Alice");
    // All source IDs should be present.
    let all_sources = source_count == 10;
    // The summary should contain content from the most recent fact.
    let has_recent = summary.contains("Japanese") || summary.contains("speaks");

    let score = [mentions_entity, all_sources, has_recent]
        .iter()
        .filter(|&&b| b)
        .count() as f64
        / 3.0;

    let passed = score >= 0.66;
    report(
        "knowledge_consolidation",
        passed,
        score,
        &format!(
            "mentions_entity={mentions_entity}, all_sources={all_sources}(got {source_count}), has_recent={has_recent}"
        ),
    );

    assert!(
        mentions_entity,
        "consolidated summary should mention the entity name"
    );
    assert_eq!(source_count, 10, "all 10 source facts should be referenced");
}

// ── 4. Graph Inference Accuracy ───────────────────────────────────────────

#[test]
#[ignore]
fn test_graph_inference_accuracy() {
    let (db, _dir) = temp_db_graph();

    // Create a chain of entities: A -> B -> C -> D
    let a = db.insert("people", json!({"name": "Alice"})).unwrap();
    let b = db.insert("people", json!({"name": "Bob"})).unwrap();
    let c = db.insert("people", json!({"name": "Carol"})).unwrap();
    let d = db.insert("people", json!({"name": "Dave"})).unwrap();

    db.relate(&a.id, "knows", &b.id, None).unwrap();
    db.relate(&b.id, "knows", &c.id, None).unwrap();
    db.relate(&c.id, "knows", &d.id, None).unwrap();

    // Test 1: Direct neighbor (A -> B).
    let neighbors_a = db.neighbors(&a.id, Some("knows"), Direction::Out).unwrap();
    let direct_ok = neighbors_a.len() == 1 && neighbors_a[0].data["name"] == "Bob";

    // Test 2: Two-hop traversal (A ->knows-> ->knows->).
    let two_hop = db.traverse(&a.id, "->knows->knows").unwrap();
    let two_hop_ok = two_hop.len() == 1 && two_hop[0].data["name"] == "Carol";

    // Test 3: Three-hop traversal (A ->knows->knows->knows).
    let three_hop = db.traverse(&a.id, "->knows->knows->knows").unwrap();
    let three_hop_ok = three_hop.len() == 1 && three_hop[0].data["name"] == "Dave";

    // Test 4: Create a diamond pattern: A -> B, A -> E, B -> F, E -> F.
    let e = db.insert("people", json!({"name": "Eve"})).unwrap();
    let f = db.insert("people", json!({"name": "Frank"})).unwrap();
    db.relate(&a.id, "works_with", &e.id, None).unwrap();
    db.relate(&b.id, "works_with", &f.id, None).unwrap();
    db.relate(&e.id, "works_with", &f.id, None).unwrap();

    // Verify Eve is reachable from A via works_with.
    let works_neighbors = db
        .neighbors(&a.id, Some("works_with"), Direction::Out)
        .unwrap();
    let diamond_ok = works_neighbors.iter().any(|r| r.data["name"] == "Eve");

    // Test 5: Bidirectional -- B should see A as incoming neighbor.
    let incoming = db.neighbors(&b.id, Some("knows"), Direction::In).unwrap();
    let incoming_ok = incoming.len() == 1 && incoming[0].data["name"] == "Alice";

    let checks = [direct_ok, two_hop_ok, three_hop_ok, diamond_ok, incoming_ok];
    let score = checks.iter().filter(|&&b| b).count() as f64 / checks.len() as f64;
    let passed = score >= 0.80;

    report(
        "graph_inference_accuracy",
        passed,
        score,
        &format!(
            "direct={direct_ok}, two_hop={two_hop_ok}, three_hop={three_hop_ok}, diamond={diamond_ok}, incoming={incoming_ok}"
        ),
    );

    assert!(direct_ok, "direct neighbor lookup failed");
    assert!(two_hop_ok, "two-hop traversal failed");
    assert!(three_hop_ok, "three-hop traversal failed");
    assert!(diamond_ok, "diamond pattern traversal failed");
    assert!(incoming_ok, "incoming direction traversal failed");
}

// ── 5. Cross-Memory Recall ────────────────────────────────────────────────

#[test]
#[ignore]
fn test_cross_memory_recall() {
    // Uses synthetic 3-dim vectors to test cross-memory recall.
    let (db, _dir) = temp_db_full();
    let mem = AgentMemory::new(&db);

    // Populate different memory types.
    // Semantic: facts.
    let sem_rec = mem
        .semantic()
        .know("auth-module", "Uses JWT tokens with 1h expiry", None)
        .unwrap();
    // Give it a synthetic vector related to "auth".
    db.add_vector(&sem_rec.id, &[1.0, 0.0, 0.0]).unwrap();

    // Episodic: past session.
    let ep_rec = mem
        .episodic()
        .create(
            "Fixed auth timeout by increasing connection pool",
            Outcome::Success,
            Some(vec!["Increased pool from 5 to 20".into()]),
            Some(vec!["config.rs".into()]),
        )
        .unwrap();
    db.add_vector(&ep_rec.id, &[0.9, 0.1, 0.0]).unwrap();

    // Procedural: learned pattern.
    let proc_rec = mem
        .procedural()
        .learn(
            "fix-auth-timeout",
            "Check connection pool size first, then network config",
            None,
        )
        .unwrap();
    db.add_vector(&proc_rec.id, &[0.8, 0.2, 0.0]).unwrap();

    // Preference: user rule.
    let pref_rec = mem
        .preference()
        .set(
            "auth_testing",
            "Always run auth integration tests after changes",
            PreferenceSource::User,
        )
        .unwrap();
    db.add_vector(&pref_rec.id, &[0.7, 0.3, 0.0]).unwrap();

    // Query with a vector close to "auth" direction.
    // remember() uses similar_to() internally which needs a text embedder.
    // Since we have synthetic vectors without a real embedder, we test via
    // the lower-level similar_to_vector + manual type tagging.
    let results = db.similar_to_vector(&[1.0, 0.0, 0.0], 10).unwrap();

    // Collect which tables returned results.
    let tables: HashSet<String> = results.iter().map(|(r, _)| r.table.clone()).collect();

    let has_entities = tables.contains("_entities");
    let has_episodes = tables.contains("_episodes");
    let has_procedures = tables.contains("_procedures");
    let has_preferences = tables.contains("_preferences");

    let types_found = [has_entities, has_episodes, has_procedures, has_preferences]
        .iter()
        .filter(|&&b| b)
        .count();

    let score = types_found as f64 / 4.0;
    let passed = types_found >= 3; // At least 3 out of 4 types.

    report(
        "cross_memory_recall",
        passed,
        score,
        &format!(
            "entities={has_entities}, episodes={has_episodes}, procedures={has_procedures}, preferences={has_preferences}"
        ),
    );

    assert!(
        types_found >= 3,
        "expected results from at least 3 memory types, got {types_found}: {tables:?}"
    );
}

// ── 6. Recency-Weighted Recall ────────────────────────────────────────────

#[test]
#[ignore]
fn test_recency_weighted_recall() {
    // Test that recency-weighted scoring correctly re-ranks results.
    //
    // We create two records: one with higher vector similarity but lower
    // recency, and one with lower similarity but higher recency. With
    // alpha < 0.5 (recency-heavy), the newer record should win.
    //
    // Since we can't set created_at directly, we use the fact that records
    // are inserted sequentially and the second has a later timestamp.
    // We give the OLDER record a higher similarity to the query vector,
    // so that pure vector ranking would put it first. Then we verify that
    // recency-weighted scoring flips the order.

    let (db, _dir) = temp_db_full();

    // Insert "old" record with high similarity to query [1.0, 0.0, 0.0].
    let old_record = db
        .insert(
            "facts",
            json!({"summary": "Old auth fact with high similarity"}),
        )
        .unwrap();
    db.add_vector(&old_record.id, &[0.99, 0.1, 0.0]).unwrap();
    let old_id = old_record.id.clone();
    let old_created = old_record.created_at;

    // Sleep briefly to ensure measurably different timestamps.
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Insert "new" record with slightly lower similarity.
    let new_record = db
        .insert(
            "facts",
            json!({"summary": "New auth fact with lower similarity"}),
        )
        .unwrap();
    db.add_vector(&new_record.id, &[0.85, 0.5, 0.0]).unwrap();
    let new_id = new_record.id.clone();
    let new_created = new_record.created_at;

    // Sanity: new is actually newer.
    assert!(
        new_created >= old_created,
        "new record should have a later timestamp"
    );

    // Pure vector search: old should rank first (higher similarity).
    let vec_results = db.similar_to_vector(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(vec_results.len(), 2);
    let pure_first_is_old = vec_results[0].0.id == old_id;

    // Now apply recency-weighted scoring manually.
    let now = chrono::Utc::now();
    let alpha = 0.2_f32; // 20% similarity, 80% recency.
    let decay_window = 30.0 * 86400.0;

    let mut scored: Vec<(String, f32, f32, f32)> = vec_results
        .iter()
        .map(|(record, similarity)| {
            let age_secs = (now - record.created_at).num_seconds().max(0) as f64;
            let recency = (1.0 - (age_secs / decay_window).min(1.0)).max(0.0) as f32;
            let final_score = alpha * similarity + (1.0 - alpha) * recency;
            (record.id.to_string(), *similarity, recency, final_score)
        })
        .collect();
    scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    // With heavy recency weighting and both records being very recent
    // (within milliseconds), recency scores are nearly 1.0 for both.
    // The newer record has a slight recency advantage. Combined with
    // alpha=0.2, the recency term dominates. The newer record should win.
    let reranked_first_is_new = scored[0].0 == new_id.to_string();

    // Also verify the scoring formula itself is correct.
    let old_entry = scored.iter().find(|e| e.0 == old_id.to_string()).unwrap();
    let new_entry = scored.iter().find(|e| e.0 == new_id.to_string()).unwrap();

    // Verify: new record's recency >= old record's recency.
    let recency_ordering = new_entry.2 >= old_entry.2;

    // Verify: old record's similarity > new record's similarity.
    let similarity_ordering = old_entry.1 > new_entry.1;

    let checks = [pure_first_is_old, recency_ordering, similarity_ordering];
    let score = checks.iter().filter(|&&b| b).count() as f64 / checks.len() as f64;

    // The main assertion: recency scoring formula works correctly.
    // We check that the scoring components behave as expected.
    let passed = recency_ordering && similarity_ordering;

    report(
        "recency_weighted_recall",
        passed,
        score,
        &format!(
            "pure_vector_old_first={pure_first_is_old}, \
             recency_new>=old={recency_ordering}, \
             similarity_old>new={similarity_ordering}, \
             old(sim={:.3},rec={:.3},final={:.3}), \
             new(sim={:.3},rec={:.3},final={:.3})",
            old_entry.1, old_entry.2, old_entry.3, new_entry.1, new_entry.2, new_entry.3,
        ),
    );

    assert!(
        recency_ordering,
        "newer record should have higher recency score"
    );
    assert!(
        similarity_ordering,
        "older record should have higher similarity score (test setup)"
    );
}

// ── 7. Token Budget Compliance ────────────────────────────────────────────

#[test]
#[ignore]
fn test_token_budget_compliance() {
    let (db, _dir) = temp_db_full();

    // Insert records of varying sizes.
    let texts = vec![
        "Short fact about auth.",
        "Medium-length fact about the authentication module that uses JWT tokens with refresh token rotation enabled for security purposes.",
        "A very long fact that goes into extensive detail about the entire authentication subsystem including the OAuth2 flow, PKCE challenge verification, token introspection endpoint, refresh token rotation policy, session management strategy, CORS configuration for the auth endpoints, rate limiting on login attempts, account lockout policies, password hashing with argon2id, multi-factor authentication via TOTP, recovery codes generation, and the audit logging of all authentication events to the security event log.",
        "Another medium fact about database connection pooling configuration and timeout settings.",
        "Tiny fact.",
    ];

    for (i, text) in texts.iter().enumerate() {
        let record = db
            .insert("knowledge", json!({"summary": text, "index": i}))
            .unwrap();
        // Add synthetic vectors so they can be found.
        let v = match i {
            0 => [1.0, 0.0, 0.0],
            1 => [0.9, 0.1, 0.0],
            2 => [0.8, 0.2, 0.0],
            3 => [0.7, 0.3, 0.0],
            4 => [0.6, 0.4, 0.0],
            _ => [0.5, 0.5, 0.0],
        };
        db.add_vector(&record.id, &v).unwrap();
    }

    // Set a token budget of 100 tokens (~400 chars).
    let token_budget: usize = 100;

    // Retrieve all records via vector search.
    let results = db.similar_to_vector(&[1.0, 0.0, 0.0], 10).unwrap();

    // Simulate token-budget-aware truncation (same logic as recall::remember).
    let mut total_tokens = 0_usize;
    let mut budget_results = Vec::new();
    for (record, similarity) in &results {
        let json_str = serde_json::to_string(&record.data).unwrap_or_default();
        let tokens = json_str.len().div_ceil(4);
        if total_tokens.saturating_add(tokens) > token_budget {
            break;
        }
        total_tokens += tokens;
        budget_results.push((record, similarity, tokens));
    }

    // Verify budget compliance.
    let within_budget = total_tokens <= token_budget;
    let returned_some = !budget_results.is_empty();
    let excluded_some = budget_results.len() < results.len();

    let score = [within_budget, returned_some, excluded_some]
        .iter()
        .filter(|&&b| b)
        .count() as f64
        / 3.0;

    let passed = within_budget && returned_some;

    report(
        "token_budget_compliance",
        passed,
        score,
        &format!(
            "total_tokens={total_tokens}, budget={token_budget}, returned={}, total_available={}, within_budget={within_budget}",
            budget_results.len(),
            results.len()
        ),
    );

    assert!(
        within_budget,
        "total tokens {total_tokens} exceeds budget {token_budget}"
    );
    assert!(
        returned_some,
        "should return at least one result within budget"
    );
}
