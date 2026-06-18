//! HTTP API server for Axil — REST endpoints over axum.
//!
//! Serves at `http://<host>:<port>/api/*` with JSON request/response.
//! Optional CORS support for browser clients.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::{HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

use axil_core::{Axil, RecordId};

/// Shared application state.
pub type AppState = Arc<Axil>;

/// Create the API router.
pub fn api_router(db: Axil) -> Router {
    let state: AppState = Arc::new(db);

    // Default to localhost-only CORS for security.
    // TODO: make configurable via axil.toml [http] allowed_origins
    const CORS_ORIGINS: &[&str] = &[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ];
    let origins: Vec<HeaderValue> = CORS_ORIGINS.iter().map(|o| o.parse().unwrap()).collect();
    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Health
        .route("/api/health", get(health))
        .route("/api/info", get(info))
        .route("/api/tables", get(tables))
        // CRUD
        .route("/api/records", post(insert_record))
        .route("/api/records", get(list_records))
        .route("/api/records/{id}", get(get_record))
        .route("/api/records/{id}", delete(delete_record))
        // Search
        .route("/api/search", post(vector_search))
        .route("/api/fts", post(fts_search))
        .route("/api/recall", post(recall))
        // Graph
        .route("/api/relate", post(create_edge))
        .route("/api/neighbors/{id}", get(neighbors))
        // Stats
        .route("/api/doctor", get(doctor))
        .route("/api/stats", get(stats))
        // Schema & autocomplete
        .route("/api/schema", get(schema))
        .with_state(state)
        .layer(cors)
}

/// Run the HTTP server.
pub async fn serve(db: Axil, host: &str, port: u16) -> anyhow::Result<()> {
    let app = api_router(db);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("Axil HTTP server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Health & Info ──────────────────────────────────────────────────────

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn info(State(db): State<AppState>) -> Result<Json<Value>, AppError> {
    let info = db.info()?;
    let files: Vec<Value> = info
        .files
        .iter()
        .map(|(p, role, size)| json!({"path": p.display().to_string(), "role": role, "size": size}))
        .collect();
    Ok(Json(json!({
        "path": info.path.display().to_string(),
        "files": files,
        "total_size": info.total_size,
        "total_records": info.total_records,
        "tables": info.tables,
        "plugins": info.plugins,
    })))
}

async fn tables(State(db): State<AppState>) -> Result<Json<Value>, AppError> {
    let tables = db.tables_with_counts()?;
    let result: Vec<Value> = tables
        .iter()
        .map(|(name, count)| json!({"table": name, "count": count}))
        .collect();
    Ok(Json(json!(result)))
}

// ── CRUD ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InsertRequest {
    table: String,
    data: Value,
}

async fn insert_record(
    State(db): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let record = db.insert(&req.table, req.data)?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": record.id.to_string(),
            "table": record.table,
            "created_at": record.created_at.to_rfc3339(),
        })),
    ))
}

async fn get_record(
    State(db): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let record_id = RecordId::from_string(&id)
        .map_err(|e| AppError::bad_request(format!("invalid ID: {e}")))?;
    match db.get(&record_id)? {
        Some(r) => Ok(Json(record_to_json(&r))),
        None => Err(AppError::not_found(format!("record not found: {id}"))),
    }
}

async fn delete_record(
    State(db): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, AppError> {
    let record_id = RecordId::from_string(&id)
        .map_err(|e| AppError::bad_request(format!("invalid ID: {e}")))?;
    match db.delete(&record_id)? {
        true => Ok(Json(json!({"deleted": true, "id": id}))),
        false => Err(AppError::not_found(format!("record not found: {id}"))),
    }
}

#[derive(Deserialize)]
struct ListParams {
    table: Option<String>,
    limit: Option<usize>,
}

async fn list_records(
    State(db): State<AppState>,
    Query(params): Query<ListParams>,
) -> Result<Json<Value>, AppError> {
    let table = params.table.as_deref().unwrap_or("_all");
    let limit = params.limit.unwrap_or(50);

    if table == "_all" {
        let tables = db.tables_with_counts()?;
        let mut all: Vec<Value> = Vec::new();
        for (tbl, _) in &tables {
            let remaining = limit.saturating_sub(all.len());
            if remaining == 0 {
                break;
            }
            if let Ok(records) = db.list_with_limit(tbl, remaining) {
                for r in &records {
                    all.push(record_to_json(r));
                }
            }
        }
        Ok(Json(json!(all)))
    } else {
        let records = db.list_with_limit(table, limit)?;
        let result: Vec<Value> = records.iter().map(record_to_json).collect();
        Ok(Json(json!(result)))
    }
}

// ── Search ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    top_k: Option<usize>,
}

async fn vector_search(
    State(db): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Value>, AppError> {
    let top_k = req.top_k.unwrap_or(5);
    let results = db.similar_to(&req.query, top_k)?;
    Ok(Json(json!(scored_results(&results))))
}

#[derive(Deserialize)]
struct FtsRequest {
    query: String,
    limit: Option<usize>,
}

async fn fts_search(
    State(db): State<AppState>,
    Json(req): Json<FtsRequest>,
) -> Result<Json<Value>, AppError> {
    let limit = req.limit.unwrap_or(10);
    let results = db.search_text(&req.query, limit)?;
    Ok(Json(json!(scored_results(&results))))
}

#[derive(Deserialize)]
struct RecallRequest {
    query: String,
    top_k: Option<usize>,
    table: Option<String>,
}

async fn recall(
    State(db): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<Value>, AppError> {
    let top_k = req.top_k.unwrap_or(5);
    // Use db.recall() for multi-signal scoring (recency, importance, graph, etc.)
    // instead of similar_to() which is pure vector search.
    match db.recall(&req.query, top_k, None) {
        Ok(results) => {
            let filtered: Vec<Value> = results
                .iter()
                .filter(|r| req.table.as_ref().map_or(true, |t| r.record.table == *t))
                .map(|r| {
                    json!({
                        "id": r.record.id.to_string(),
                        "table": r.record.table,
                        "data": r.record.data,
                        "score": r.score,
                        "created_at": r.record.created_at.to_rfc3339(),
                        "updated_at": r.record.updated_at.to_rfc3339(),
                    })
                })
                .collect();
            Ok(Json(json!(filtered)))
        }
        Err(_) => {
            // Fall back to vector search if recall isn't available.
            let results = db.similar_to(&req.query, top_k)?;
            let filtered: Vec<_> = results
                .iter()
                .filter(|(r, _)| req.table.as_ref().map_or(true, |t| r.table == *t))
                .cloned()
                .collect();
            Ok(Json(json!(scored_results(&filtered))))
        }
    }
}

// ── Graph ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RelateRequest {
    from: String,
    edge_type: String,
    to: String,
    props: Option<Value>,
}

async fn create_edge(
    State(db): State<AppState>,
    Json(req): Json<RelateRequest>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let from = RecordId::from_string(&req.from)
        .map_err(|e| AppError::bad_request(format!("invalid 'from' ID: {e}")))?;
    let to = RecordId::from_string(&req.to)
        .map_err(|e| AppError::bad_request(format!("invalid 'to' ID: {e}")))?;
    let edge_id = db.relate(&from, &req.edge_type, &to, req.props)?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "edge_id": edge_id.to_string(),
            "from": req.from,
            "edge_type": req.edge_type,
            "to": req.to,
        })),
    ))
}

#[derive(Deserialize)]
struct NeighborParams {
    #[serde(rename = "type")]
    edge_type: Option<String>,
}

async fn neighbors(
    State(db): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<NeighborParams>,
) -> Result<Json<Value>, AppError> {
    let record_id = RecordId::from_string(&id)
        .map_err(|e| AppError::bad_request(format!("invalid ID: {e}")))?;
    let results = db.neighbors(
        &record_id,
        params.edge_type.as_deref(),
        axil_core::Direction::Both,
    )?;
    let items: Vec<Value> = results.iter().map(record_to_json).collect();
    Ok(Json(json!(items)))
}

// ── Stats/Doctor ──────────────────────────────────────────────────────

async fn doctor(State(db): State<AppState>) -> Result<Json<Value>, AppError> {
    let report = db.doctor()?;
    Ok(Json(serde_json::to_value(&report)?))
}

async fn stats(State(db): State<AppState>) -> Result<Json<Value>, AppError> {
    let db_stats = db.stats(None)?;
    Ok(Json(serde_json::to_value(&db_stats)?))
}

// ── Schema & autocomplete ────────────────────────────────────

async fn schema(State(db): State<AppState>) -> Result<Json<Value>, AppError> {
    let tables = db.tables_with_counts()?;
    let table_names: Vec<&str> = tables.iter().map(|(n, _)| n.as_str()).collect();

    // Collect field names from a sample of records.
    let mut field_names = std::collections::BTreeSet::new();
    for (table, _) in tables.iter().take(10) {
        if let Ok(records) = db.list(table) {
            for record in records.iter().take(5) {
                if let Some(obj) = record.data.as_object() {
                    for key in obj.keys() {
                        field_names.insert(format!("data.{}", key));
                    }
                }
            }
        }
    }
    field_names.insert("table".into());
    field_names.insert("created_at".into());
    field_names.insert("updated_at".into());

    let syntax = axil_ql::syntax_metadata();

    Ok(Json(json!({
        "tables": table_names,
        "fields": field_names,
        "syntax": syntax,
        "plugins": {
            "vector": db.has_vector_index(),
            "graph": db.has_graph_index(),
        },
    })))
}

// ── Helpers ───────────────────────────────────────────────────────────

fn scored_results(results: &[(axil_core::Record, f32)]) -> Vec<Value> {
    results
        .iter()
        .map(|(r, score)| {
            let mut j = record_to_json(r);
            j["score"] = json!(score);
            j
        })
        .collect()
}

fn record_to_json(r: &axil_core::Record) -> Value {
    json!({
        "id": r.id.to_string(),
        "table": r.table,
        "data": r.data,
        "created_at": r.created_at.to_rfc3339(),
        "updated_at": r.updated_at.to_rfc3339(),
    })
}

// ── Error handling ────────────────────────────────────────────────────

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

impl From<axil_core::AxilError> for AppError {
    fn from(e: axil_core::AxilError) -> Self {
        match &e {
            axil_core::AxilError::NotFound(_) => AppError {
                status: StatusCode::NOT_FOUND,
                message: e.to_string(),
            },
            axil_core::AxilError::InvalidQuery(_) => AppError {
                status: StatusCode::BAD_REQUEST,
                message: e.to_string(),
            },
            _ => AppError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: e.to_string(),
            },
        }
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_app() -> (Router, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        (api_router(db), dir)
    }

    async fn body_json(body: Body) -> Value {
        let bytes = body.collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let (app, _dir) = test_app();
        let req = axum::http::Request::builder()
            .uri("/api/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn get_invalid_id_returns_400() {
        let (app, _dir) = test_app();
        let req = axum::http::Request::builder()
            .uri("/api/records/not_a_valid_id")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_nonexistent_record_returns_404() {
        let (app, _dir) = test_app();
        // Use a valid ULID format that doesn't exist
        let req = axum::http::Request::builder()
            .uri("/api/records/01000000000000000000000000")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn insert_and_get_record() {
        let (app, _dir) = test_app();
        // Insert
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/api/records")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"table":"notes","data":{"text":"hello"}}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_json(resp.into_body()).await;
        let id = json["id"].as_str().unwrap().to_string();

        // Get
        let req = axum::http::Request::builder()
            .uri(format!("/api/records/{id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        assert_eq!(json["data"]["text"], "hello");
    }

    #[tokio::test]
    async fn insert_missing_table_returns_400() {
        let (app, _dir) = test_app();
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/api/records")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"data":{"text":"no table"}}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Missing required "table" field should be 400 or 422
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
        );
    }

    #[tokio::test]
    async fn delete_nonexistent_record_returns_404() {
        let (app, _dir) = test_app();
        let req = axum::http::Request::builder()
            .method("DELETE")
            .uri("/api/records/01000000000000000000000000")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn tables_returns_array() {
        let (app, _dir) = test_app();
        let req = axum::http::Request::builder()
            .uri("/api/tables")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp.into_body()).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let (app, _dir) = test_app();
        let req = axum::http::Request::builder()
            .uri("/api/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
