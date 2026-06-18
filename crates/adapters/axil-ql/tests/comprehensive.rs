//! Comprehensive test suite for AxilQL.
//!
//! Covers: parser (30+ queries), compiler, error messages, edge cases, fuzz safety.

use axil_ql::{ast::*, parse};

// ═══════════════════════════════════════════════════════════════════════
// Parser tests — all keyword combinations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn p01_recall_basic() {
    let q = parse(r#"RECALL "auth timeout bug" TOP 10"#).unwrap();
    assert!(matches!(&q, Query::Recall { text, top_k: 10, .. } if text == "auth timeout bug"));
}

#[test]
fn p02_recall_with_from() {
    let q = parse(r#"RECALL "auth timeout" TOP 5 FROM sessions"#).unwrap();
    match &q {
        Query::Recall {
            top_k: 5, clauses, ..
        } => {
            assert!(matches!(&clauses[0], Clause::From(t) if t == "sessions"));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p03_recall_with_traverse() {
    let q = parse(r#"RECALL "auth error" TOP 5 TRAVERSE ->mentions"#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::Traverse(p) if p == "->mentions"));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p04_recall_with_where() {
    let q = parse(
        r#"RECALL "auth error" TOP 5 WHERE table = "sessions" AND created_at > "2026-03-01""#,
    )
    .unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            if let Clause::Where(conds) = &clauses[0] {
                assert_eq!(conds.len(), 2);
                assert_eq!(conds[0].field, "table");
                assert_eq!(conds[0].op, CompareOp::Eq);
                assert_eq!(conds[1].field, "created_at");
                assert_eq!(conds[1].op, CompareOp::Gt);
            } else {
                panic!("expected Where clause");
            }
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p05_recall_combined_traverse_where() {
    let q = parse(r#"RECALL "auth error" TOP 5 TRAVERSE ->mentions WHERE table = "sessions" AND created_at > "2026-03-01""#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::Traverse(_)));
            assert!(matches!(&clauses[1], Clause::Where(_)));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p06_recall_boost_recency() {
    let q = parse(r#"RECALL "deployment issue" TOP 10 BOOST recency 0.4"#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            assert!(
                matches!(&clauses[0], Clause::Boost(BoostType::Recency, w) if (*w - 0.4).abs() < 0.01)
            );
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p07_recall_boost_graph() {
    let q = parse(r#"RECALL "test" TOP 5 BOOST graph 0.8"#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::Boost(BoostType::Graph, _)));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p08_recall_boost_feedback() {
    let q = parse(r#"RECALL "test" TOP 5 BOOST feedback 1"#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::Boost(BoostType::Feedback, _)));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p09_recall_profile() {
    let q = parse(r#"RECALL "memory leak" TOP 10 TRAVERSE ->mentions PROFILE"#).unwrap();
    assert!(q.has_profile());
}

#[test]
fn p10_find_basic() {
    let q = parse(r#"FIND "authentication""#).unwrap();
    match &q {
        Query::Find { text, field, .. } => {
            assert_eq!(text, "authentication");
            assert!(field.is_none());
        }
        _ => panic!("expected Find"),
    }
}

#[test]
fn p11_find_with_in() {
    let q = parse(r#"FIND "authentication" IN summary"#).unwrap();
    match &q {
        Query::Find { text, field, .. } => {
            assert_eq!(text, "authentication");
            assert_eq!(field.as_deref(), Some("summary"));
        }
        _ => panic!("expected Find"),
    }
}

#[test]
fn p12_find_with_from_order_limit_offset() {
    let q = parse(r#"FIND "error" FROM logs ORDER BY created_at DESC LIMIT 25 OFFSET 50"#).unwrap();
    match &q {
        Query::Find { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::From(t) if t == "logs"));
            assert!(matches!(&clauses[1], Clause::OrderBy(f, SortDir::Desc) if f == "created_at"));
            assert!(matches!(&clauses[2], Clause::Limit(25)));
            assert!(matches!(&clauses[3], Clause::Offset(50)));
        }
        _ => panic!("expected Find"),
    }
}

#[test]
fn p13_traverse_basic() {
    let q = parse("TRAVERSE ->modified->file FROM rec_01HZ3ABC").unwrap();
    match &q {
        Query::Traverse { path, from, .. } => {
            assert_eq!(path, "->modified->file");
            assert_eq!(from.as_deref(), Some("rec_01HZ3ABC"));
        }
        _ => panic!("expected Traverse"),
    }
}

#[test]
fn p14_traverse_inbound() {
    let q = parse("TRAVERSE <-created_by FROM users").unwrap();
    match &q {
        Query::Traverse { path, from, .. } => {
            assert_eq!(path, "<-created_by");
            assert_eq!(from.as_deref(), Some("users"));
        }
        _ => panic!("expected Traverse"),
    }
}

#[test]
fn p15_traverse_bidirectional() {
    let q = parse("TRAVERSE <->related FROM nodes").unwrap();
    match &q {
        Query::Traverse { path, from, .. } => {
            assert_eq!(path, "<->related");
            assert_eq!(from.as_deref(), Some("nodes"));
        }
        _ => panic!("expected Traverse"),
    }
}

#[test]
fn p16_traverse_requires_from() {
    let err = parse("TRAVERSE ->edge").unwrap_err();
    assert!(err.message.contains("FROM"));
}

#[test]
fn p17_get() {
    let q = parse("GET my_record_id").unwrap();
    assert!(matches!(&q, Query::Get { id } if id == "my_record_id"));
}

#[test]
fn p18_get_quoted() {
    let q = parse(r#"GET "01ABCDEF12345678901234AB""#).unwrap();
    assert!(matches!(&q, Query::Get { id } if id == "01ABCDEF12345678901234AB"));
}

#[test]
fn p19_count_from_table() {
    let q = parse("COUNT FROM sessions").unwrap();
    assert!(matches!(&q, Query::Count { table } if table.as_deref() == Some("sessions")));
}

#[test]
fn p20_count_all() {
    let q = parse("COUNT").unwrap();
    assert!(matches!(&q, Query::Count { table } if table.is_none()));
}

#[test]
fn p21_explain_recall() {
    let q = parse(r#"EXPLAIN RECALL "x" TOP 5"#).unwrap();
    match &q {
        Query::Explain { inner } => {
            assert!(matches!(inner.as_ref(), Query::Recall { .. }));
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn p22_explain_traverse() {
    let q = parse("EXPLAIN TRAVERSE ->edge FROM t").unwrap();
    assert!(matches!(&q, Query::Explain { .. }));
}

#[test]
fn p23_where_all_operators() {
    let q = parse(
        r#"RECALL "x" TOP 1 WHERE a = 1 AND b != 2 AND c > 3 AND d < 4 AND e >= 5 AND f <= 6"#,
    )
    .unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            if let Clause::Where(conds) = &clauses[0] {
                assert_eq!(conds.len(), 6);
                assert_eq!(conds[0].op, CompareOp::Eq);
                assert_eq!(conds[1].op, CompareOp::Ne);
                assert_eq!(conds[2].op, CompareOp::Gt);
                assert_eq!(conds[3].op, CompareOp::Lt);
                assert_eq!(conds[4].op, CompareOp::Gte);
                assert_eq!(conds[5].op, CompareOp::Lte);
            } else {
                panic!("expected Where");
            }
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p24_where_contains() {
    let q = parse(r#"RECALL "x" TOP 1 WHERE tags CONTAINS "rust""#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            if let Clause::Where(conds) = &clauses[0] {
                assert_eq!(conds[0].op, CompareOp::Contains);
            } else {
                panic!("expected Where");
            }
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p25_where_bool_null_values() {
    let q = parse(r#"RECALL "x" TOP 1 WHERE active = true AND deleted = false AND meta = null"#)
        .unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            if let Clause::Where(conds) = &clauses[0] {
                assert_eq!(conds[0].value, ConditionValue::Bool(true));
                assert_eq!(conds[1].value, ConditionValue::Bool(false));
                assert_eq!(conds[2].value, ConditionValue::Null);
            } else {
                panic!("expected Where");
            }
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p26_where_float_value() {
    let q = parse(r#"RECALL "x" TOP 1 WHERE score > 0.95"#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            if let Clause::Where(conds) = &clauses[0] {
                assert!(
                    matches!(&conds[0].value, ConditionValue::Float(f) if (*f - 0.95).abs() < 0.001)
                );
            } else {
                panic!("expected Where");
            }
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p27_case_insensitive_all_keywords() {
    // All keywords should be case-insensitive
    let q =
        parse(r#"recall "x" top 5 from t where a = 1 order by a asc limit 10 offset 0 profile"#)
            .unwrap();
    assert!(matches!(&q, Query::Recall { .. }));
    assert!(q.has_profile());
}

#[test]
fn p28_single_quoted_strings() {
    let q = parse("RECALL 'hello world' TOP 5").unwrap();
    match &q {
        Query::Recall { text, .. } => assert_eq!(text, "hello world"),
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p29_comment_handling() {
    let q = parse("-- search for auth bugs\nRECALL \"auth\" TOP 5").unwrap();
    assert!(matches!(&q, Query::Recall { .. }));
}

#[test]
fn p30_multiline_query() {
    let q = parse("RECALL \"test\"\n  TOP 10\n  FROM sessions\n  WHERE x = 1").unwrap();
    match &q {
        Query::Recall {
            top_k: 10, clauses, ..
        } => {
            assert!(matches!(&clauses[0], Clause::From(t) if t == "sessions"));
            assert!(matches!(&clauses[1], Clause::Where(_)));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p31_escaped_string() {
    let q = parse(r#"RECALL "hello \"world\"" TOP 5"#).unwrap();
    match &q {
        Query::Recall { text, .. } => assert_eq!(text, r#"hello "world""#),
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p32_dotted_field_name() {
    let q = parse(r#"RECALL "x" TOP 1 WHERE data.name = "test""#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            if let Clause::Where(conds) = &clauses[0] {
                assert_eq!(conds[0].field, "data.name");
            } else {
                panic!("expected Where");
            }
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p33_order_by_default_asc() {
    let q = parse(r#"FIND "x" ORDER BY name"#).unwrap();
    match &q {
        Query::Find { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::OrderBy(f, SortDir::Asc) if f == "name"));
        }
        _ => panic!("expected Find"),
    }
}

#[test]
fn p34_multiple_traversals() {
    let q = parse(r#"RECALL "x" TOP 5 TRAVERSE ->edge1 TRAVERSE ->edge2"#).unwrap();
    match &q {
        Query::Recall { clauses, .. } => {
            assert!(matches!(&clauses[0], Clause::Traverse(p) if p == "->edge1"));
            assert!(matches!(&clauses[1], Clause::Traverse(p) if p == "->edge2"));
        }
        _ => panic!("expected Recall"),
    }
}

#[test]
fn p35_complex_combined_query() {
    let q = parse(r#"RECALL "deployment issue" TOP 10 FROM prod_logs TRAVERSE ->mentions WHERE severity = "critical" AND team = "platform" BOOST recency 0.4 ORDER BY created_at DESC LIMIT 5 OFFSET 0 PROFILE"#).unwrap();
    match &q {
        Query::Recall {
            text,
            top_k: 10,
            clauses,
        } => {
            assert_eq!(text, "deployment issue");
            assert!(clauses.len() >= 5);
            assert!(q.has_profile());
        }
        _ => panic!("expected Recall"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Error tests — malformed queries produce helpful messages
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn e01_empty_query() {
    let err = parse("").unwrap_err();
    assert!(err.message.contains("expected a query keyword"));
}

#[test]
fn e02_missing_text_after_recall() {
    let err = parse("RECALL TOP 5").unwrap_err();
    assert!(err.message.contains("quoted string"));
}

#[test]
fn e03_missing_top_after_recall() {
    let err = parse(r#"RECALL "test""#).unwrap_err();
    assert!(err.message.contains("TOP"));
}

#[test]
fn e04_zero_top_k() {
    let err = parse(r#"RECALL "test" TOP 0"#).unwrap_err();
    assert!(err.message.contains("not positive"));
}

#[test]
fn e05_negative_top_k() {
    // -5 is lexed as a negative integer
    let err = parse(r#"RECALL "test" TOP -5"#).unwrap_err();
    assert!(!err.message.is_empty());
}

#[test]
fn e06_invalid_boost_type() {
    let err = parse(r#"RECALL "test" TOP 5 BOOST magic 0.5"#).unwrap_err();
    assert!(err.message.contains("unknown boost type"));
    assert!(err.suggestion.is_some());
}

#[test]
fn e07_select_suggestion() {
    let err = parse("SELECT FROM users").unwrap_err();
    assert!(err.suggestion.is_some());
    assert!(err.suggestion.as_ref().unwrap().contains("RECALL"));
}

#[test]
fn e08_insert_suggestion() {
    let err = parse("INSERT INTO sessions").unwrap_err();
    assert!(err.suggestion.is_some());
    assert!(err.suggestion.as_ref().unwrap().contains("read-only"));
}

#[test]
fn e09_unterminated_string() {
    let err = parse(r#"RECALL "unterminated"#).unwrap_err();
    assert!(err.message.contains("unterminated"));
}

#[test]
fn e10_unexpected_character() {
    let err = parse("RECALL @foo TOP 5").unwrap_err();
    assert!(err.message.contains("unexpected character"));
}

#[test]
fn e11_missing_order_by() {
    let err = parse(r#"FIND "x" ORDER name"#).unwrap_err();
    assert!(err.message.contains("BY"));
}

#[test]
fn e12_error_position_info() {
    let err = parse("RECALL TOP").unwrap_err();
    assert_eq!(err.span.line, 1);
    assert!(err.span.column > 1); // points to TOP, not start of line
}

#[test]
fn e13_trailing_garbage() {
    let err = parse(r#"GET some_id GARBAGE"#).unwrap_err();
    assert!(err.message.contains("unexpected"));
}

// ═══════════════════════════════════════════════════════════════════════
// Compiler tests — integration with Axil database
// ═══════════════════════════════════════════════════════════════════════

fn setup_db() -> (tempfile::TempDir, axil_core::Axil) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.axil");
    let db = axil_core::Axil::open(&db_path).build().unwrap();
    (dir, db)
}

#[test]
fn c01_count_empty() {
    let (_dir, db) = setup_db();
    let result = axil_ql::run(&db, "COUNT").unwrap();
    assert_eq!(result.count, 0);
}

#[test]
fn c02_count_with_data() {
    let (_dir, db) = setup_db();
    db.insert("t", serde_json::json!({"a": 1})).unwrap();
    db.insert("t", serde_json::json!({"a": 2})).unwrap();
    db.insert("u", serde_json::json!({"b": 3})).unwrap();

    let result = axil_ql::run(&db, "COUNT FROM t").unwrap();
    assert_eq!(result.count, 2);

    let result = axil_ql::run(&db, "COUNT").unwrap();
    assert_eq!(result.count, 3);
}

#[test]
fn c03_get_existing() {
    let (_dir, db) = setup_db();
    let r = db.insert("t", serde_json::json!({"msg": "hello"})).unwrap();
    let id = r.id.to_string();

    let result = axil_ql::run(&db, &format!(r#"GET "{id}""#)).unwrap();
    assert_eq!(result.count, 1);
    assert_eq!(result.results[0]["data"]["msg"], "hello");
}

#[test]
fn c04_get_missing() {
    let (_dir, db) = setup_db();
    let fake = axil_core::RecordId::new();
    let result = axil_ql::run(&db, &format!(r#"GET "{fake}""#)).unwrap();
    assert_eq!(result.count, 0);
}

#[test]
fn c05_explain_returns_plan() {
    let (_dir, db) = setup_db();
    let result = axil_ql::run(&db, r#"EXPLAIN RECALL "test" TOP 5"#).unwrap();
    assert!(result.plan.is_some());
    assert_eq!(result.count, 0); // EXPLAIN doesn't execute
}

#[test]
fn c06_explain_count() {
    let (_dir, db) = setup_db();
    let result = axil_ql::run(&db, "EXPLAIN COUNT FROM t").unwrap();
    assert!(result.plan.is_some());
}

#[test]
fn c07_explain_get() {
    let (_dir, db) = setup_db();
    let result = axil_ql::run(&db, r#"EXPLAIN GET "01ABCDEFGHHJKMNPQRSTVWX""#).unwrap();
    assert!(result.plan.is_some());
}

#[test]
fn c08_error_response_format() {
    let err = axil_ql::run(&setup_db().1, "INVALID QUERY").unwrap_err();
    let resp = axil_ql::ErrorResponse::from(&err);
    assert!(!resp.error.is_empty());
    // Parse errors should have position info
    assert!(resp.line.is_some());
}

#[test]
fn c09_elapsed_ms_present() {
    let (_dir, db) = setup_db();
    let result = axil_ql::run(&db, "COUNT").unwrap();
    assert!(result.elapsed_ms >= 0.0);
}

// ═══════════════════════════════════════════════════════════════════════
// Fuzz safety — random/adversarial input must never panic
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn f01_empty_string() {
    let _ = parse("");
}

#[test]
fn f02_only_whitespace() {
    let _ = parse("   \n\t  ");
}

#[test]
fn f03_only_comments() {
    let _ = parse("-- just a comment\n-- another");
}

#[test]
fn f04_garbage_bytes() {
    let _ = parse("!@#$%^&*()");
}

#[test]
fn f05_very_long_string() {
    let long = "a".repeat(100_000);
    let _ = parse(&format!(r#"RECALL "{long}" TOP 5"#));
}

#[test]
fn f06_deeply_nested_comments() {
    let input = "-- c1\n-- c2\n-- c3\nRECALL \"x\" TOP 1";
    let _ = parse(input);
}

#[test]
fn f07_repeated_keywords() {
    let _ = parse("RECALL RECALL RECALL TOP TOP TOP");
}

#[test]
fn f08_max_integer() {
    let _ = parse(&format!("RECALL \"x\" TOP {}", i64::MAX));
}

#[test]
fn f09_unicode_string() {
    let q = parse(r#"RECALL "日本語テスト 🎉" TOP 5"#).unwrap();
    match &q {
        Query::Recall { text, .. } => assert!(text.contains("日本語")),
        _ => panic!("expected Recall"),
    }
}

#[test]
fn f10_null_bytes_in_string() {
    // Null bytes shouldn't crash the lexer
    let _ = parse("RECALL \"hello\x00world\" TOP 5");
}

// ═══════════════════════════════════════════════════════════════════════
// AST utility tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ast_has_profile() {
    let q = parse(r#"RECALL "x" TOP 5 PROFILE"#).unwrap();
    assert!(q.has_profile());

    let q = parse(r#"RECALL "x" TOP 5"#).unwrap();
    assert!(!q.has_profile());
}

#[test]
fn ast_clauses_empty_for_get() {
    let q = parse("GET some_id").unwrap();
    assert!(q.clauses().is_empty());
}

#[test]
fn ast_clauses_empty_for_count() {
    let q = parse("COUNT").unwrap();
    assert!(q.clauses().is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Serialization tests — AST and QueryResult serialize cleanly
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn serialize_ast() {
    let q = parse(r#"RECALL "test" TOP 5 FROM logs WHERE x = 1"#).unwrap();
    let json = serde_json::to_string(&q).unwrap();
    assert!(json.contains("Recall"));
    assert!(json.contains("test"));
}

#[test]
fn serialize_query_result() {
    let (_dir, db) = setup_db();
    let result = axil_ql::run(&db, "COUNT").unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("count"));
    assert!(json.contains("elapsed_ms"));
}
