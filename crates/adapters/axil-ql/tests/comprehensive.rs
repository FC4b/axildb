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
    assert!(matches!(&q, Query::Count { table, .. } if table.as_deref() == Some("sessions")));
}

#[test]
fn p20_count_all() {
    let q = parse("COUNT").unwrap();
    assert!(matches!(&q, Query::Count { table, .. } if table.is_none()));
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

#[test]
fn f11_escaped_multibyte_char_in_string() {
    // A backslash before a multi-byte character used to skip one byte into
    // it and panic on the next slice (found by the nightly `ql_parse` fuzz).
    // Unknown escapes keep the backslash, and the character must survive
    // intact rather than being decoded byte-wise.
    let q = parse(r#"RECALL "a\Ɨb" TOP 5"#).unwrap();
    match &q {
        Query::Recall { text, .. } => assert_eq!(text, r"a\Ɨb"),
        _ => panic!("expected Recall"),
    }
    let _ = parse("RECALL \"\\🎉\" TOP 5");
    let _ = parse("RECALL \"unterminated \\Ɨ");
}

#[test]
fn f12_fuzz_crash_escape_before_multibyte_char() {
    // The exact input from the nightly fuzz crash
    // (crash-2dcac7589eb9d7cbd4bda4f71aa199586aaa009e).
    let mut bytes = vec![
        77, 67, 39, 50, 34, 127, 127, 127, 127, 83, 0, 0, 0, 127, 127, 198, 151, 198, 148,
    ];
    bytes.extend(std::iter::repeat(92).take(109));
    bytes.extend([198, 151, 198, 148]);
    let input = std::str::from_utf8(&bytes).expect("fuzz input is valid UTF-8");
    let _ = parse(input);
}

#[test]
fn f13_unexpected_multibyte_char_is_reported_whole() {
    let err = parse("RECALL Ɨ").unwrap_err();
    assert!(
        err.to_string().contains('Ɨ'),
        "the error names the character, not its first byte: {err}"
    );
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

// ═══════════════════════════════════════════════════════════════════════
// KNOWN AT — knowledge-time filtering on TRAVERSE
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn p60_traverse_known_at_parses() {
    let q =
        parse("TRAVERSE ->depends_on FROM rec_01HZ3ABC KNOWN AT '2026-01-01T00:00:00Z' LIMIT 5")
            .unwrap();
    match &q {
        Query::TraverseKnownAt {
            path,
            from,
            known_at,
            clauses,
            ..
        } => {
            assert_eq!(path, "->depends_on");
            assert_eq!(from, "rec_01HZ3ABC");
            assert_eq!(known_at, "2026-01-01T00:00:00Z");
            assert!(matches!(clauses.as_slice(), [Clause::Limit(5)]));
        }
        _ => panic!("expected TraverseKnownAt, got {q:?}"),
    }
    assert_eq!(q.clauses(), &[Clause::Limit(5)]);

    // A seed that is itself spelled like the keyword stays a seed.
    let q = parse("TRAVERSE ->e FROM known KNOWN AT '2026-01-01T00:00:00Z'").unwrap();
    assert!(matches!(&q, Query::TraverseKnownAt { from, .. } if from == "known"));
}

#[test]
fn p63_plain_traverse_keeps_published_shape() {
    // Destructured without `..` on purpose: this stops compiling if a field is
    // ever added to the published `Query::Traverse` variant, which
    // cargo-semver-checks treats as a major break.
    let q = parse("TRAVERSE ->e FROM rec_01 LIMIT 3").unwrap();
    match q {
        Query::Traverse {
            path,
            from,
            clauses,
        } => {
            assert_eq!(path, "->e");
            assert_eq!(from.as_deref(), Some("rec_01"));
            assert_eq!(clauses, vec![Clause::Limit(3)]);
        }
        other => panic!("expected Traverse, got {other:?}"),
    }
}

#[test]
fn serialize_traverse_known_at_roundtrips() {
    let q = parse("TRAVERSE ->e FROM rec_01 KNOWN AT '2026-01-01T00:00:00Z'").unwrap();
    let json = serde_json::to_value(&q).unwrap();
    assert_eq!(json["TraverseKnownAt"]["known_at"], "2026-01-01T00:00:00Z");
    let back: Query = serde_json::from_value(json).unwrap();
    assert_eq!(back, q);

    // Plain TRAVERSE serialises exactly as the published shape did.
    let plain = serde_json::to_value(parse("TRAVERSE ->e FROM rec_01").unwrap()).unwrap();
    assert_eq!(
        plain,
        serde_json::json!({"Traverse": {"path": "->e", "from": "rec_01", "clauses": []}})
    );
}

#[test]
fn p61_traverse_known_at_validates_timestamp() {
    let err = parse("TRAVERSE ->edge FROM rec_01 KNOWN AT 'not-a-date'").unwrap_err();
    assert!(err.message.contains("RFC 3339"), "{}", err.message);
}

#[test]
fn p62_traverse_known_requires_at() {
    let err = parse("TRAVERSE ->edge FROM rec_01 KNOWN").unwrap_err();
    assert!(err.message.contains("KNOWN requires AT"), "{}", err.message);
}

#[test]
fn c60_traverse_known_at_filters_by_edge_creation_time() {
    use axil_graph::AxilBuilderGraphExt;

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("kt.axil");
    let db = axil_core::Axil::open(&db_path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();

    let a = db
        .insert("services", serde_json::json!({"name": "api"}))
        .unwrap();
    let b = db
        .insert("services", serde_json::json!({"name": "db"}))
        .unwrap();
    db.relate(&a.id, "depends_on", &b.id, None).unwrap();

    // No cutoff: the edge is traversable.
    let q = format!(r#"TRAVERSE ->depends_on FROM "{}""#, a.id);
    assert_eq!(axil_ql::run(&db, &q).unwrap().count, 1);

    // Knowledge cutoff before the edge was recorded: filtered out.
    let q = format!(
        r#"TRAVERSE ->depends_on FROM "{}" KNOWN AT '2020-01-01T00:00:00Z'"#,
        a.id
    );
    assert_eq!(axil_ql::run(&db, &q).unwrap().count, 0);

    // Cutoff after the edge was recorded: visible again.
    let q = format!(
        r#"TRAVERSE ->depends_on FROM "{}" KNOWN AT '2030-01-01T00:00:00Z'"#,
        a.id
    );
    assert_eq!(axil_ql::run(&db, &q).unwrap().count, 1);
}

// ═══════════════════════════════════════════════════════════════════════
// Soft keywords — AGG / GROUP / KNOWN / AT stay usable as identifiers
// ═══════════════════════════════════════════════════════════════════════

/// Field name of the first condition of a statement's WHERE, wherever the
/// statement keeps it.
fn first_where_field(q: &Query) -> String {
    match q {
        Query::Count {
            where_conditions, ..
        }
        | Query::Agg {
            where_conditions, ..
        } => where_conditions[0].field.clone(),
        other => match other.clauses().iter().find_map(|c| match c {
            Clause::Where(conds) => Some(conds[0].field.clone()),
            _ => None,
        }) {
            Some(f) => f,
            None => panic!("no WHERE in {other:?}"),
        },
    }
}

#[test]
fn p70_soft_keywords_as_where_fields() {
    for name in ["at", "known", "agg", "group", "AT", "Known"] {
        for q in [
            format!(r#"COUNT FROM ev WHERE {name} > "2026-01-01""#),
            format!(r#"COUNT FROM ev WHERE x = 1 AND {name} = 2"#),
            format!(r#"RECALL "x" TOP 5 WHERE {name} = 1"#),
            format!(r#"FIND "x" WHERE {name} CONTAINS "y""#),
            format!(r#"AGG count FROM ev WHERE {name} = 1 GROUP BY family"#),
            format!(r#"TRAVERSE ->e FROM t WHERE {name} = 1"#),
        ] {
            let parsed = parse(&q).unwrap_or_else(|e| panic!("{q}: {e}"));
            let field = if q.contains(" AND ") {
                match &parsed {
                    Query::Count {
                        where_conditions, ..
                    } => where_conditions[1].field.clone(),
                    _ => unreachable!(),
                }
            } else {
                first_where_field(&parsed)
            };
            // Identifiers keep the spelling the user wrote.
            assert_eq!(field, name, "{q}");
        }
    }
}

#[test]
fn p71_soft_keywords_as_table_names() {
    for name in ["at", "known", "agg", "group"] {
        let q = parse(&format!("COUNT FROM {name}")).unwrap();
        assert!(matches!(&q, Query::Count { table: Some(t), .. } if t == name));

        let q = parse(&format!(r#"RECALL "x" TOP 5 FROM {name}"#)).unwrap();
        assert!(matches!(&q.clauses()[0], Clause::From(t) if t == name));

        let q = parse(&format!("AGG count FROM {name}")).unwrap();
        assert!(matches!(&q, Query::Agg { table, .. } if table == name));

        let q = parse(&format!("TRAVERSE ->e FROM {name}")).unwrap();
        assert!(matches!(&q, Query::Traverse { from: Some(f), .. } if f == name));
    }
}

#[test]
fn p72_soft_keywords_in_order_by_in_and_agg_fields() {
    for name in ["at", "known", "agg", "group"] {
        let q = parse(&format!(r#"FIND "x" ORDER BY {name} DESC"#)).unwrap();
        assert!(
            matches!(&q.clauses()[0], Clause::OrderBy(f, SortDir::Desc) if f == name),
            "{q:?}"
        );

        let q = parse(&format!(r#"FIND "x" IN {name}"#)).unwrap();
        assert!(matches!(&q, Query::Find { field: Some(f), .. } if f == name));

        let q = parse(&format!(
            "AGG avg({name}), max({name}) FROM t GROUP BY {name}"
        ))
        .unwrap();
        match &q {
            Query::Agg {
                metrics, group_by, ..
            } => {
                assert_eq!(
                    metrics,
                    &vec![
                        AggSpec::Avg(name.to_string()),
                        AggSpec::Max(name.to_string())
                    ]
                );
                assert_eq!(group_by.as_deref(), Some(name));
            }
            _ => panic!("expected Agg"),
        }

        // A bare soft keyword is also a bare string value, like any identifier.
        let q = parse(&format!("COUNT FROM t WHERE kind = {name}")).unwrap();
        match &q {
            Query::Count {
                where_conditions, ..
            } => assert_eq!(
                where_conditions[0].value,
                ConditionValue::String(name.to_string())
            ),
            _ => panic!("expected Count"),
        }
    }
}

#[test]
fn p73_soft_keywords_still_work_in_keyword_position() {
    // Statement-leading AGG and GROUP BY, any case.
    let q = parse("agg count from group where at > 1 group by group").unwrap();
    match &q {
        Query::Agg {
            table,
            where_conditions,
            group_by,
            ..
        } => {
            assert_eq!(table, "group");
            assert_eq!(where_conditions[0].field, "at");
            assert_eq!(group_by.as_deref(), Some("group"));
        }
        _ => panic!("expected Agg"),
    }
    assert!(matches!(
        parse("EXPLAIN AGG count FROM t GROUP BY g").unwrap(),
        Query::Explain { .. }
    ));
    // GROUP must still be followed by BY.
    assert!(parse("AGG count FROM t GROUP g").is_err());
}

#[test]
fn c70_where_on_field_named_at_executes() {
    let (_dir, db) = setup_db();
    db.insert("ev", serde_json::json!({"at": "2025-06-01T00:00:00Z"}))
        .unwrap();
    db.insert("ev", serde_json::json!({"at": "2026-06-01T00:00:00Z"}))
        .unwrap();
    db.insert("ev", serde_json::json!({"at": "2026-07-01T00:00:00Z"}))
        .unwrap();

    let result = axil_ql::run(&db, r#"COUNT FROM ev WHERE at > "2026-01-01""#).unwrap();
    assert_eq!(result.count, 2);
}

// ═══════════════════════════════════════════════════════════════════════
// KNOWN AT on a table-seeded TRAVERSE
// ═══════════════════════════════════════════════════════════════════════

fn setup_graph_db() -> (tempfile::TempDir, axil_core::Axil) {
    use axil_graph::AxilBuilderGraphExt;

    let dir = tempfile::tempdir().unwrap();
    let db = axil_core::Axil::open(dir.path().join("kt.axil"))
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (dir, db)
}

/// Endpoint ids of a query result, in result order.
fn result_ids(db: &axil_core::Axil, q: &str) -> Vec<String> {
    axil_ql::run(db, q)
        .unwrap_or_else(|e| panic!("{q}: {e}"))
        .results
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

/// Two seeds in `svc`, three endpoints in `lib`. `s1->l1` and `s2->l2` are
/// recorded before the returned cutoff; `s1->l3` and `s2->l1` after it.
fn seed_edges_around_cutoff(db: &axil_core::Axil) -> String {
    use serde_json::json;

    let s1 = db.insert("svc", json!({"name": "s1", "tier": 1})).unwrap();
    let s2 = db.insert("svc", json!({"name": "s2", "tier": 2})).unwrap();
    let l1 = db.insert("lib", json!({"name": "l1", "rank": 3})).unwrap();
    let l2 = db.insert("lib", json!({"name": "l2", "rank": 1})).unwrap();
    let l3 = db.insert("lib", json!({"name": "l3", "rank": 2})).unwrap();

    db.relate(&s1.id, "uses", &l1.id, None).unwrap();
    db.relate(&s2.id, "uses", &l2.id, None).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let cutoff = chrono::Utc::now().to_rfc3339();
    std::thread::sleep(std::time::Duration::from_millis(20));
    db.relate(&s1.id, "uses", &l3.id, None).unwrap();
    db.relate(&s2.id, "uses", &l1.id, None).unwrap();
    cutoff
}

#[test]
fn c61_traverse_from_table_known_at_applies_cutoff() {
    let (_dir, db) = setup_graph_db();
    let cutoff = seed_edges_around_cutoff(&db);

    let names = |q: &str| -> Vec<String> {
        let mut v: Vec<String> = axil_ql::run(&db, q)
            .unwrap_or_else(|e| panic!("{q}: {e}"))
            .results
            .iter()
            .map(|r| r["data"]["name"].as_str().unwrap().to_string())
            .collect();
        v.sort();
        v
    };

    // Today's graph: every endpoint, deduplicated across seeds.
    assert_eq!(names("TRAVERSE ->uses FROM svc"), ["l1", "l2", "l3"]);
    // The graph as it was known at the cutoff: only the early edges.
    assert_eq!(
        names(&format!("TRAVERSE ->uses FROM svc KNOWN AT '{cutoff}'")),
        ["l1", "l2"]
    );
    // Before any edge was recorded: nothing.
    assert!(names("TRAVERSE ->uses FROM svc KNOWN AT '2020-01-01T00:00:00Z'").is_empty());
    // WHERE still selects the seed rows: s1 only reached l1 by the cutoff.
    assert_eq!(
        names(&format!(
            "TRAVERSE ->uses FROM svc KNOWN AT '{cutoff}' WHERE tier = 1"
        )),
        ["l1"]
    );
}

#[test]
fn c62_known_at_after_every_edge_matches_no_cutoff() {
    let (_dir, db) = setup_graph_db();
    seed_edges_around_cutoff(&db);
    // A lone seed with more endpoints than the default result cap.
    let hub = db
        .insert("hub", serde_json::json!({"name": "hub"}))
        .unwrap();
    for i in 0..105 {
        let leaf = db.insert("leaf", serde_json::json!({"i": i})).unwrap();
        db.relate(&hub.id, "has", &leaf.id, None).unwrap();
    }

    let future = "KNOWN AT '2999-01-01T00:00:00Z'";
    for (path, table, suffix) in [
        ("->uses", "svc", ""),
        ("->uses", "svc", "WHERE tier = 1"),
        ("->uses", "svc", "ORDER BY rank DESC"),
        ("->uses", "svc", "ORDER BY rank OFFSET 1"),
        ("->uses", "svc", "WHERE tier = 2 ORDER BY rank DESC LIMIT 5"),
        ("->uses", "svc", "LIMIT 2 OFFSET 1"),
        // No LIMIT: both stop at the same default result cap.
        ("->has", "hub", ""),
        ("->has", "hub", "LIMIT 3 OFFSET 2"),
        ("->has", "hub", "ORDER BY i DESC LIMIT 200"),
    ] {
        let plain = format!("TRAVERSE {path} FROM {table} {suffix}");
        let cut = format!("TRAVERSE {path} FROM {table} {future} {suffix}");
        let expected = result_ids(&db, &plain);
        assert!(!expected.is_empty(), "{plain}");
        assert_eq!(result_ids(&db, &cut), expected, "{cut}");
    }
    assert_eq!(result_ids(&db, "TRAVERSE ->has FROM hub").len(), 100);
}

#[test]
fn c64_traverse_from_table_known_at_orders_before_paging() {
    let (_dir, db) = setup_graph_db();
    let hub = db
        .insert("hub", serde_json::json!({"name": "hub"}))
        .unwrap();
    for i in 0..10 {
        let leaf = db.insert("leaf", serde_json::json!({"i": i})).unwrap();
        db.relate(&hub.id, "has", &leaf.id, None).unwrap();
    }
    // ORDER BY ranks every endpoint before OFFSET/LIMIT page them, so this is
    // the true top of the ordering, not a sorted sample of the first few
    // endpoints the walk happened to reach.
    let result = axil_ql::run(
        &db,
        "TRAVERSE ->has FROM hub KNOWN AT '2999-01-01T00:00:00Z' ORDER BY i DESC LIMIT 3 OFFSET 1",
    )
    .unwrap();
    let got: Vec<i64> = result
        .results
        .iter()
        .map(|r| r["data"]["i"].as_i64().unwrap())
        .collect();
    assert_eq!(got, [8, 7, 6]);
}

#[test]
fn c65_known_at_rejects_chained_traverse() {
    let (_dir, db) = setup_graph_db();
    let seed = db.insert("svc", serde_json::json!({"name": "s"})).unwrap();
    for from in [seed.id.to_string(), "svc".to_string()] {
        for prefix in ["", "EXPLAIN "] {
            let q = format!(
                r#"{prefix}TRAVERSE ->a FROM "{from}" KNOWN AT '2026-01-01T00:00:00Z' TRAVERSE ->b"#
            );
            let err = axil_ql::run(&db, &q).unwrap_err().to_string();
            assert!(err.contains("chained TRAVERSE"), "{q}: {err}");
        }
    }
}

#[test]
fn c63_explain_traverse_known_at_shows_cutoff() {
    let (_dir, db) = setup_graph_db();
    let seed = db.insert("svc", serde_json::json!({"name": "s"})).unwrap();
    let ts = "2026-01-01T00:00:00Z";

    for from in [seed.id.to_string(), "svc".to_string()] {
        let q = format!(r#"EXPLAIN TRAVERSE ->uses FROM "{from}" KNOWN AT '{ts}'"#);
        let plan = axil_ql::run(&db, &q).unwrap().plan.expect("plan");
        let step = plan
            .plan
            .iter()
            .find(|s| s.step_type == "graph_traverse")
            .unwrap_or_else(|| panic!("{q}: no graph_traverse step"));
        assert_eq!(step.params["known_at"], ts, "{q}");

        // Without KNOWN AT the plan carries no cutoff.
        let q = format!(r#"EXPLAIN TRAVERSE ->uses FROM "{from}""#);
        let plan = axil_ql::run(&db, &q).unwrap().plan.expect("plan");
        let step = plan
            .plan
            .iter()
            .find(|s| s.step_type == "graph_traverse")
            .unwrap();
        assert!(step.params.get("known_at").is_none(), "{q}");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// COUNT / AGG see every matching row, not the first page
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn c80_count_and_agg_are_not_capped_at_a_page() {
    let (_dir, db) = setup_db();
    for v in 1..=130 {
        db.insert("runs", serde_json::json!({"v": v, "kind": "a"}))
            .unwrap();
    }

    assert_eq!(axil_ql::run(&db, "COUNT FROM runs").unwrap().count, 130);
    assert_eq!(
        axil_ql::run(&db, "COUNT FROM runs WHERE v > 0")
            .unwrap()
            .count,
        130
    );
    assert_eq!(axil_ql::run(&db, "COUNT WHERE v > 0").unwrap().count, 130);

    let out = axil_ql::run(&db, "AGG count, sum(v) FROM runs WHERE v > 0").unwrap();
    assert_eq!(out.results[0]["total_rows"], 130);
    assert_eq!(out.results[0]["groups"][0]["sum_v"], 8515.0);

    let out = axil_ql::run(&db, "AGG count, max(v) FROM runs GROUP BY kind").unwrap();
    assert_eq!(out.results[0]["groups"][0]["count"], 130);
    assert_eq!(out.results[0]["groups"][0]["max_v"], 130.0);
}
