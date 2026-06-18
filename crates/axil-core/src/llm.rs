//! LLM provider interface — optional intelligence upgrade.
//!
//! Every feature works at Level 0 (no LLM). LLM is a quality boost, not a
//! dependency. The `LlmProvider` trait allows Rust library users to plug in
//! any LLM backend. For CLI/agent users, the agent itself IS the LLM
//! (Path A: CLI + Skill).

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::error::{AxilError, Result};

// ── LlmProvider trait ───────────────────────────────────────────────────────

/// Trait for plugging in an LLM backend.
///
/// Implementations must be thread-safe (`Send + Sync`). All methods may
/// block on network I/O — callers should avoid holding locks when invoking
/// them.
pub trait LlmProvider: Send + Sync {
    /// Free-form text completion.
    fn complete(&self, prompt: &str) -> Result<LlmResponse>;

    /// Structured extraction — LLM returns JSON matching the schema hint.
    ///
    /// The `schema_hint` is a human-readable description of the expected
    /// JSON structure (not a JSON Schema document). Implementations should
    /// include it in the system prompt.
    fn extract_json(&self, prompt: &str, schema_hint: &str) -> Result<LlmResponse>;

    /// Model name for logging and debugging.
    fn model_name(&self) -> &str;

    /// Whether this provider is available (connected, has API key, etc.)
    fn is_available(&self) -> bool;
}

/// Response from an LLM call, including token usage for cost tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text.
    pub text: String,
    /// Input tokens consumed (0 if unknown).
    pub input_tokens: u64,
    /// Output tokens generated (0 if unknown).
    pub output_tokens: u64,
}

// ── NoLlm (default) ────────────────────────────────────────────────────────

/// Default provider when no LLM is configured. Always returns an error
/// so callers fall back to algorithmic paths.
pub struct NoLlm;

impl LlmProvider for NoLlm {
    fn complete(&self, _prompt: &str) -> Result<LlmResponse> {
        Err(AxilError::plugin("no LLM configured"))
    }

    fn extract_json(&self, _prompt: &str, _schema_hint: &str) -> Result<LlmResponse> {
        Err(AxilError::plugin("no LLM configured"))
    }

    fn model_name(&self) -> &str {
        "none"
    }

    fn is_available(&self) -> bool {
        false
    }
}

// ── Cost tracking ───────────────────────────────────────────────────────────

/// Session-level LLM usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmUsage {
    /// Total LLM calls made.
    pub calls: u64,
    /// Total input tokens consumed.
    pub input_tokens: u64,
    /// Total output tokens generated.
    pub output_tokens: u64,
    /// Estimated cost in USD (based on model pricing).
    pub estimated_cost_usd: f64,
    /// Number of calls that fell back to algorithmic.
    pub fallback_count: u64,
}

/// Tracks LLM usage across a session with thread-safe atomic counters.
pub struct LlmUsageTracker {
    calls: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    fallback_count: AtomicU64,
    /// Cost per 1M input tokens in USD. 0.0 if unknown.
    cost_per_1m_input: f64,
    /// Cost per 1M output tokens in USD. 0.0 if unknown.
    cost_per_1m_output: f64,
}

impl LlmUsageTracker {
    /// Create a new tracker with optional pricing info.
    pub fn new(cost_per_1m_input: f64, cost_per_1m_output: f64) -> Self {
        Self {
            calls: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
            cost_per_1m_input,
            cost_per_1m_output,
        }
    }

    /// Record a successful LLM call.
    pub fn record_call(&self, input_tokens: u64, output_tokens: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.input_tokens.fetch_add(input_tokens, Ordering::Relaxed);
        self.output_tokens
            .fetch_add(output_tokens, Ordering::Relaxed);
    }

    /// Record a fallback to algorithmic (LLM was unavailable or budget exceeded).
    pub fn record_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current usage snapshot.
    pub fn usage(&self) -> LlmUsage {
        let input = self.input_tokens.load(Ordering::Relaxed);
        let output = self.output_tokens.load(Ordering::Relaxed);
        let cost = (input as f64 * self.cost_per_1m_input
            + output as f64 * self.cost_per_1m_output)
            / 1_000_000.0;
        LlmUsage {
            calls: self.calls.load(Ordering::Relaxed),
            input_tokens: input,
            output_tokens: output,
            estimated_cost_usd: cost,
            fallback_count: self.fallback_count.load(Ordering::Relaxed),
        }
    }
}

// ── Budget limits ───────────────────────────────────────────────────────────

/// Configurable limits for LLM usage. When any limit is reached, the system
/// falls back to algorithmic — never hard-fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmLimits {
    /// Maximum LLM calls per minute (0 = unlimited).
    pub max_calls_per_minute: u64,
    /// Maximum total tokens per session (0 = unlimited).
    pub max_tokens_per_session: u64,
    /// Maximum daily budget in USD (0.0 = unlimited).
    pub budget_usd_per_day: f64,
}

impl Default for LlmLimits {
    fn default() -> Self {
        Self {
            max_calls_per_minute: 10,
            max_tokens_per_session: 50_000,
            budget_usd_per_day: 1.0,
        }
    }
}

/// Rate limiter that tracks per-minute call counts.
pub struct LlmRateLimiter {
    limits: LlmLimits,
    /// Timestamps of recent calls (for per-minute rate limiting).
    recent_calls: Mutex<Vec<std::time::Instant>>,
}

impl LlmRateLimiter {
    /// Create a new rate limiter with the given limits.
    pub fn new(limits: LlmLimits) -> Self {
        Self {
            limits,
            recent_calls: Mutex::new(Vec::new()),
        }
    }

    /// Atomically check whether an LLM call is allowed and, if so, record it.
    ///
    /// Returns `true` if the call should proceed, `false` if it should
    /// fall back to algorithmic. Combines check + record in a single lock
    /// acquisition to avoid TOCTOU races under concurrent access.
    pub fn check_and_record(&self, usage: &LlmUsage) -> bool {
        // Check session token limit.
        if self.limits.max_tokens_per_session > 0
            && (usage.input_tokens + usage.output_tokens) >= self.limits.max_tokens_per_session
        {
            return false;
        }

        // Check daily budget.
        if self.limits.budget_usd_per_day > 0.0
            && usage.estimated_cost_usd >= self.limits.budget_usd_per_day
        {
            return false;
        }

        // Check per-minute rate and record atomically.
        if self.limits.max_calls_per_minute > 0 {
            let now = std::time::Instant::now();
            let one_minute_ago = now - std::time::Duration::from_secs(60);
            if let Ok(mut calls) = self.recent_calls.lock() {
                calls.retain(|t| *t > one_minute_ago);
                if calls.len() as u64 >= self.limits.max_calls_per_minute {
                    return false;
                }
                calls.push(now);
            }
        } else if let Ok(mut calls) = self.recent_calls.lock() {
            calls.push(std::time::Instant::now());
        }

        true
    }

    /// Get the current limits.
    pub fn limits(&self) -> &LlmLimits {
        &self.limits
    }
}

// ── LLM configuration ──────────────────────────────────────────────────────

/// Configuration for the LLM provider, loadable from `axil.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// LLM API endpoint (e.g. "https://api.openai.com/v1/chat/completions").
    pub endpoint: Option<String>,
    /// Model identifier (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514").
    pub model: Option<String>,
    /// API key (prefer env var AXIL_LLM_API_KEY over config file).
    pub api_key: Option<String>,
    /// Cost limits.
    pub limits: LlmLimits,
    /// Cost per 1M input tokens in USD (for cost tracking).
    pub cost_per_1m_input: f64,
    /// Cost per 1M output tokens in USD (for cost tracking).
    pub cost_per_1m_output: f64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            model: None,
            api_key: None,
            limits: LlmLimits::default(),
            cost_per_1m_input: 0.15, // Conservative default (GPT-4o-mini level)
            cost_per_1m_output: 0.60,
        }
    }
}

impl LlmConfig {
    /// Check if enough configuration exists to create an HTTP provider.
    pub fn is_configured(&self) -> bool {
        self.endpoint.is_some() && self.model.is_some() && self.resolved_api_key().is_some()
    }

    /// Resolve API key: env var takes precedence over config.
    pub fn resolved_api_key(&self) -> Option<String> {
        std::env::var("AXIL_LLM_API_KEY")
            .ok()
            .or_else(|| self.api_key.clone())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_llm_is_unavailable() {
        let provider = NoLlm;
        assert!(!provider.is_available());
        assert_eq!(provider.model_name(), "none");
        assert!(provider.complete("test").is_err());
        assert!(provider.extract_json("test", "{}").is_err());
    }

    #[test]
    fn usage_tracker_records() {
        let tracker = LlmUsageTracker::new(0.15, 0.60);
        tracker.record_call(100, 50);
        tracker.record_call(200, 100);
        tracker.record_fallback();

        let usage = tracker.usage();
        assert_eq!(usage.calls, 2);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
        assert_eq!(usage.fallback_count, 1);
        assert!(usage.estimated_cost_usd > 0.0);
    }

    #[test]
    fn rate_limiter_token_limit() {
        let limits = LlmLimits {
            max_calls_per_minute: 0,
            max_tokens_per_session: 100,
            budget_usd_per_day: 0.0,
        };
        let limiter = LlmRateLimiter::new(limits);

        let under_limit = LlmUsage {
            calls: 1,
            input_tokens: 50,
            output_tokens: 40,
            estimated_cost_usd: 0.0,
            fallback_count: 0,
        };
        assert!(limiter.check_and_record(&under_limit));

        let at_limit = LlmUsage {
            calls: 2,
            input_tokens: 60,
            output_tokens: 50,
            estimated_cost_usd: 0.0,
            fallback_count: 0,
        };
        assert!(!limiter.check_and_record(&at_limit));
    }

    #[test]
    fn rate_limiter_budget_limit() {
        let limits = LlmLimits {
            max_calls_per_minute: 0,
            max_tokens_per_session: 0,
            budget_usd_per_day: 1.0,
        };
        let limiter = LlmRateLimiter::new(limits);

        let under = LlmUsage {
            calls: 1,
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.50,
            fallback_count: 0,
        };
        assert!(limiter.check_and_record(&under));

        let over = LlmUsage {
            calls: 10,
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 1.50,
            fallback_count: 0,
        };
        assert!(!limiter.check_and_record(&over));
    }

    #[test]
    fn llm_config_default_not_configured() {
        let config = LlmConfig::default();
        assert!(!config.is_configured());
    }

    #[test]
    fn llm_response_serialization() {
        let resp = LlmResponse {
            text: "hello".to_string(),
            input_tokens: 10,
            output_tokens: 5,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: LlmResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, "hello");
        assert_eq!(back.input_tokens, 10);
    }
}
