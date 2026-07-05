//! [`SyncClient`] — speaks the Atlas sync protocol over HTTPS.
//!
//! The client mirror of the server's unified `/v1/sync` surface. Every call
//! carries the caller's Personal Access Token as a bearer credential; the
//! server derives the tenant from it. Behind the default-off `http` feature so
//! an offline/OSS build pulls no HTTP client.

use std::time::Duration;

use axil_atlas_proto::{
    BootstrapSnapshot, CompoundQuery, CompoundResult, Locator, PullQuery, PullResponse, PushBatch,
    PushResponse, Tier,
};

use crate::SyncError;

/// A thin HTTP client for one Atlas endpoint + PAT.
pub struct SyncClient {
    base: String,
    token: String,
    agent: ureq::Agent,
}

impl SyncClient {
    /// Build a client for `endpoint` (e.g. `https://atlas.example.com`)
    /// authenticating with `token` (an `atlas_pat_...` PAT).
    pub fn new(endpoint: impl Into<String>, token: impl Into<String>) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(10)))
            .build()
            .new_agent();
        Self {
            base: endpoint.into().trim_end_matches('/').to_string(),
            token: token.into(),
            agent,
        }
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }

    /// `POST /v1/sync/push` — the write path.
    pub fn push(&self, batch: &PushBatch) -> Result<PushResponse, SyncError> {
        let body = serde_json::to_string(batch)?;
        let mut resp = self
            .agent
            .post(format!("{}/v1/sync/push", self.base))
            .header("Authorization", &self.bearer())
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
            .map_err(transport)?;
        let text = resp.body_mut().read_to_string().map_err(transport)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// `GET /v1/sync/pull` — the read/merge path.
    pub fn pull(&self, q: &PullQuery) -> Result<PullResponse, SyncError> {
        let mut url = format!(
            "{}/v1/sync/pull?member={}&since={}&limit={}&tier={}",
            self.base,
            enc(&q.member),
            q.since,
            q.limit,
            tier_str(q.tier)
        );
        if let Some(t) = &q.table {
            url.push_str(&format!("&table={}", enc(t)));
        }
        let mut resp = self
            .agent
            .get(url)
            .header("Authorization", &self.bearer())
            .call()
            .map_err(transport)?;
        let text = resp.body_mut().read_to_string().map_err(transport)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// `POST /v1/compound/{topic}` — the cross-project payoff. `topic` is the
    /// human label; the client-embedded query vector rides in the body.
    pub fn compound(&self, topic: &str, q: &CompoundQuery) -> Result<CompoundResult, SyncError> {
        let body = serde_json::to_string(q)?;
        let mut resp = self
            .agent
            .post(format!("{}/v1/compound/{}", self.base, enc(topic)))
            .header("Authorization", &self.bearer())
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
            .map_err(transport)?;
        let text = resp.body_mut().read_to_string().map_err(transport)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// `GET /v1/canonical/lookup/{id}` — resolve a concept across projects.
    pub fn canonical_lookup(&self, canonical_id: &str) -> Result<Vec<Locator>, SyncError> {
        let mut resp = self
            .agent
            .get(format!("{}/v1/canonical/lookup/{}", self.base, enc(canonical_id)))
            .header("Authorization", &self.bearer())
            .call()
            .map_err(transport)?;
        let text = resp.body_mut().read_to_string().map_err(transport)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let locators = v
            .get("locators")
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| serde_json::from_value(x.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        Ok(locators)
    }

    /// `GET /v1/sync/bootstrap` — snapshot the current distillate so a new or
    /// long-offline device skips replaying the op-log from zero.
    pub fn bootstrap(&self, member: &str) -> Result<BootstrapSnapshot, SyncError> {
        let mut resp = self
            .agent
            .get(format!("{}/v1/sync/bootstrap?member={}", self.base, enc(member)))
            .header("Authorization", &self.bearer())
            .call()
            .map_err(transport)?;
        let text = resp.body_mut().read_to_string().map_err(transport)?;
        Ok(serde_json::from_str(&text)?)
    }
}

fn transport(e: ureq::Error) -> SyncError {
    SyncError::Transport(e.to_string())
}

fn tier_str(t: Tier) -> &'static str {
    match t {
        Tier::Distillate => "distillate",
        Tier::Raw => "raw",
    }
}

/// Minimal percent-encoder for a single query/path segment.
fn enc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
