//! Path B — the built-in HTTP doc fetcher, behind the default-off
//! `web-docs` feature.
//!
//! Offline-first is the default posture: nothing here is compiled unless
//! `web-docs` is explicitly enabled, and even then it is used only to
//! fill gaps local extraction cannot.
//!
//! npm is fetched from `registry.npmjs.org`, whose package JSON carries
//! the full README as clean markdown. Cargo is deliberately *not*
//! fetched over the web — the registry source cache (Path 0) is present
//! after any `cargo build`, Path A (`deps ingest`) covers the rare gap,
//! and scraping rendered HTML doc sites is an explicit non-goal.

use crate::manifest::{Dependency, Ecosystem};

/// Fetch documentation for a dependency over HTTP.
///
/// Returns `None` when the ecosystem has no web source wired, the
/// request fails, or the response carries no usable doc text.
pub fn fetch_web_doc(dep: &Dependency) -> Option<String> {
    match dep.ecosystem {
        Ecosystem::Npm => fetch_npm_readme(&dep.name),
        // Cargo / Python / Go / Java: the on-disk cache (Path 0) covers
        // them; see the module docs. Path B is npm-only for now.
        Ecosystem::Cargo | Ecosystem::Python | Ecosystem::Go | Ecosystem::Java => None,
    }
}

/// Fetch a package's README from the npm registry.
fn fetch_npm_readme(name: &str) -> Option<String> {
    let url = format!("https://registry.npmjs.org/{name}");
    let body = http_get(&url)?;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    let readme = json.get("readme").and_then(|v| v.as_str())?.trim();
    if readme.is_empty() {
        None
    } else {
        Some(readme.to_string())
    }
}

/// GET a URL and return the response body as a string.
fn http_get(url: &str) -> Option<String> {
    use std::io::Read;
    let response = ureq::get(url).call().ok()?;
    let mut body = String::new();
    response
        .into_body()
        .as_reader()
        .read_to_string(&mut body)
        .ok()?;
    Some(body)
}
