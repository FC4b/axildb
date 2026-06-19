//! Minimal HTTP **Adapter** — serves `GET /recall?q=...&k=...` as JSON.
//!
//! This is the Tier-3 `Adapter` reference the extensibility docs point at: it
//! proves the public surface is sufficient to put Axil behind a new protocol
//! **out of tree**, using only `axil_core`'s public API + the standard library
//! (no web framework, no async runtime). A third party writes exactly this much.
//!
//! Run:   `cargo run --example http_adapter`
//! Query: `curl 'http://127.0.0.1:8080/recall?q=auth&k=3'`

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use axil_core::{Adapter, AxilError, Protocol, Result};

/// An `Adapter` that serves recall over HTTP/1.1.
struct HttpAdapter {
    addr: String,
    db: Option<Arc<axil_core::Axil>>,
}

impl Adapter for HttpAdapter {
    fn id(&self) -> &str {
        "http-example"
    }

    fn protocol(&self) -> Protocol {
        Protocol::Http
    }

    fn bind(&mut self, db: Arc<axil_core::Axil>) -> Result<()> {
        // The Adapter takes a shared handle so several adapters can run against
        // the same database. Here we just stash it for `run`.
        self.db = Some(db);
        Ok(())
    }

    fn run(self) -> Result<()> {
        let db = self
            .db
            .ok_or_else(|| AxilError::plugin("run() called before bind()"))?;
        let listener = TcpListener::bind(&self.addr)
            .map_err(|e| AxilError::plugin(format!("bind {}: {e}", self.addr)))?;
        eprintln!(
            "http-example adapter listening on http://{} — GET /recall?q=...&k=...",
            self.addr
        );
        for stream in listener.incoming().flatten() {
            // One request per connection (Connection: close). A real adapter
            // would pool/thread; this stays minimal on purpose.
            handle(stream, &db);
        }
        Ok(())
    }
}

fn handle(mut stream: TcpStream, db: &axil_core::Axil) {
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // "GET /recall?q=auth&k=3 HTTP/1.1"
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");

    let body = match route(db, path) {
        Ok(json) => json,
        Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
}

/// Route a request path to a JSON response using only the public `Axil` API.
fn route(db: &axil_core::Axil, path: &str) -> Result<String> {
    let Some(query) = path.strip_prefix("/recall") else {
        return Ok(serde_json::json!({ "routes": ["/recall?q=<query>&k=<n>"] }).to_string());
    };
    let query = query.strip_prefix('?').unwrap_or("");

    let mut q = String::new();
    let mut k = 5usize;
    for pair in query.split('&') {
        match pair.split_once('=') {
            Some(("q", v)) => q = url_decode(v),
            Some(("k", v)) => k = v.parse().unwrap_or(5),
            _ => {}
        }
    }

    // Recall always returns a valid (possibly empty) result set. Rich semantic
    // hits need a backend: build with `--features embed` and embed record fields.
    let hits: Vec<_> = db
        .recall(&q, k, None)?
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "id": r.record.id.to_string(),
                "table": r.record.table,
                "score": r.score,
            })
        })
        .collect();
    Ok(serde_json::json!({ "query": q, "hits": hits }).to_string())
}

/// Minimal percent/`+` decoding — enough for the demo query string.
fn url_decode(s: &str) -> String {
    s.replace('+', " ").replace("%20", " ")
}

fn main() -> Result<()> {
    // Seed a self-contained temp database so the example runs standalone.
    let dir = tempfile::tempdir().map_err(|e| AxilError::plugin(e.to_string()))?;
    let db = axil_core::Axil::open(dir.path().join("http-demo.axil")).build()?;
    db.insert(
        "decisions",
        serde_json::json!({ "summary": "Chose redb for ACID single-file storage" }),
    )?;
    db.insert(
        "errors",
        serde_json::json!({ "error": "auth timeout on login", "fix": "raised the DB pool size" }),
    )?;

    let mut adapter = HttpAdapter {
        addr: "127.0.0.1:8080".to_string(),
        db: None,
    };
    adapter.bind(Arc::new(db))?;
    adapter.run()
}
