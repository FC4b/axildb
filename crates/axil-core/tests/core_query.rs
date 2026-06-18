use axil_core::{Axil, Op, SortDirection};
use serde_json::json;

fn setup_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("query_test.axil");
    let db = Axil::open(&path).build().unwrap();

    db.insert(
        "products",
        json!({"name": "Alpha", "price": 10, "category": "tools"}),
    )
    .unwrap();
    db.insert(
        "products",
        json!({"name": "Beta", "price": 25, "category": "tools"}),
    )
    .unwrap();
    db.insert(
        "products",
        json!({"name": "Gamma", "price": 50, "category": "parts"}),
    )
    .unwrap();
    db.insert(
        "products",
        json!({"name": "Delta", "price": 75, "category": "parts"}),
    )
    .unwrap();
    db.insert(
        "products",
        json!({"name": "Epsilon", "price": 100, "category": "premium"}),
    )
    .unwrap();

    (db, dir)
}

#[test]
fn query_eq_filter() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .where_field("category", Op::Eq, json!("tools"))
        .exec()
        .unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn query_range_filter() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .where_field("price", Op::Gte, json!(25))
        .where_field("price", Op::Lt, json!(100))
        .exec()
        .unwrap();
    assert_eq!(results.len(), 3); // Beta(25), Gamma(50), Delta(75)
}

#[test]
fn query_order_by_asc() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .order_by("price", SortDirection::Asc)
        .exec()
        .unwrap();
    let prices: Vec<i64> = results
        .iter()
        .map(|r| r.data["price"].as_i64().unwrap())
        .collect();
    assert_eq!(prices, vec![10, 25, 50, 75, 100]);
}

#[test]
fn query_order_by_desc() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .order_by("price", SortDirection::Desc)
        .exec()
        .unwrap();
    let prices: Vec<i64> = results
        .iter()
        .map(|r| r.data["price"].as_i64().unwrap())
        .collect();
    assert_eq!(prices, vec![100, 75, 50, 25, 10]);
}

#[test]
fn query_pagination() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .order_by("price", SortDirection::Asc)
        .limit(2)
        .offset(1)
        .exec()
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].data["name"], "Beta");
    assert_eq!(results[1].data["name"], "Gamma");
}

#[test]
fn query_contains() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .where_field("name", Op::Contains, json!("a"))
        .exec()
        .unwrap();
    // Alpha, Gamma, Delta, Epsilon — names containing "a"
    assert_eq!(results.len(), 4);
}

#[test]
fn query_combined_filter_sort_page() {
    let (db, _dir) = setup_db();
    let results = db
        .query()
        .table("products")
        .where_field("price", Op::Gt, json!(10))
        .order_by("price", SortDirection::Desc)
        .limit(2)
        .exec()
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].data["name"], "Epsilon");
    assert_eq!(results[1].data["name"], "Delta");
}

#[test]
fn query_empty_table() {
    let (db, _dir) = setup_db();
    let results = db.query().table("nonexistent").exec().unwrap();
    assert!(results.is_empty());
}
