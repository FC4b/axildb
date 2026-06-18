//! HTTP-based LLM provider for any OpenAI-compatible API.
//!
//! Gated behind the `llm-http` feature flag to avoid pulling in HTTP
//! dependencies for users who don't need them.
//!
//! Works with: OpenAI, Anthropic (via compatible endpoint), Ollama, OpenRouter,
//! and any other provider that speaks the OpenAI chat completions format.

use serde_json::json;

use crate::error::{AxilError, Result};
use crate::llm::{LlmProvider, LlmResponse};

/// Generic HTTP provider for OpenAI-compatible chat completion APIs.
pub struct HttpLlm {
    /// API endpoint URL (e.g. "https://api.openai.com/v1/chat/completions").
    endpoint: String,
    /// API key for authentication.
    api_key: String,
    /// Model identifier (e.g. "gpt-4o-mini").
    model: String,
    /// Reusable HTTP agent (connection pool + TLS state).
    agent: ureq::Agent,
    /// Maximum retries on transient failure.
    max_retries: u32,
}

impl HttpLlm {
    /// Create a new HTTP LLM provider.
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build(),
        );
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            model: model.into(),
            agent,
            max_retries: 2,
        }
    }

    /// Set request timeout in seconds.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(secs)))
                .build(),
        );
        self
    }

    /// Set maximum retry count for transient failures.
    pub fn with_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Create from an `LlmConfig`, returning `None` if not configured.
    pub fn from_config(config: &crate::llm::LlmConfig) -> Option<Self> {
        let endpoint = config.endpoint.as_ref()?;
        let model = config.model.as_ref()?;
        let api_key = config.resolved_api_key()?;
        Some(Self::new(endpoint, api_key, model))
    }

    /// Make the actual HTTP request, with retry logic.
    fn call(&self, messages: &[serde_json::Value]) -> Result<(String, u64, u64)> {
        let body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        });

        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Simple backoff: 1s, 2s
                std::thread::sleep(std::time::Duration::from_secs(attempt as u64));
            }

            match self.do_request(&body) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Don't retry on auth errors or bad requests.
                    let err_str = format!("{e}");
                    if err_str.contains("401") || err_str.contains("403") || err_str.contains("400")
                    {
                        return Err(e);
                    }
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| AxilError::plugin("LLM request failed")))
    }

    fn do_request(&self, body: &serde_json::Value) -> Result<(String, u64, u64)> {
        let body_str = serde_json::to_string(body)
            .map_err(|e| AxilError::plugin(format!("failed to serialize request: {e}")))?;

        let mut response = self
            .agent
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .send(body_str.as_bytes())
            .map_err(|e| AxilError::plugin(format!("LLM HTTP request failed: {e}")))?;

        let body_text = response
            .body_mut()
            .read_to_string()
            .map_err(|e| AxilError::plugin(format!("failed to read LLM response: {e}")))?;

        let response_body: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| AxilError::plugin(format!("failed to parse LLM response: {e}")))?;

        // Extract the response text from OpenAI-compatible format.
        let text = response_body
            .pointer("/choices/0/message/content")
            .and_then(|v: &serde_json::Value| v.as_str())
            .ok_or_else(|| {
                AxilError::plugin(format!(
                    "unexpected LLM response format: {}",
                    serde_json::to_string(&response_body).unwrap_or_default()
                ))
            })?
            .to_string();

        // Extract token usage if available.
        let input_tokens = response_body
            .pointer("/usage/prompt_tokens")
            .and_then(|v: &serde_json::Value| v.as_u64())
            .unwrap_or(0);
        let output_tokens = response_body
            .pointer("/usage/completion_tokens")
            .and_then(|v: &serde_json::Value| v.as_u64())
            .unwrap_or(0);

        Ok((text, input_tokens, output_tokens))
    }
}

impl LlmProvider for HttpLlm {
    fn complete(&self, prompt: &str) -> Result<LlmResponse> {
        let messages = vec![json!({
            "role": "user",
            "content": prompt,
        })];

        let (text, input_tokens, output_tokens) = self.call(&messages)?;
        Ok(LlmResponse {
            text,
            input_tokens,
            output_tokens,
        })
    }

    fn extract_json(&self, prompt: &str, schema_hint: &str) -> Result<LlmResponse> {
        let messages = vec![
            json!({
                "role": "system",
                "content": format!(
                    "You are a structured data extractor. \
                     Return ONLY valid JSON matching this schema: {schema_hint}\n\
                     No markdown, no explanation, just the JSON."
                ),
            }),
            json!({
                "role": "user",
                "content": prompt,
            }),
        ];

        let (text, input_tokens, output_tokens) = self.call(&messages)?;

        // Strip markdown code fences if present.
        let cleaned = text
            .trim()
            .strip_prefix("```json")
            .or_else(|| text.trim().strip_prefix("```"))
            .unwrap_or(text.trim())
            .strip_suffix("```")
            .unwrap_or(text.trim())
            .trim()
            .to_string();

        Ok(LlmResponse {
            text: cleaned,
            input_tokens,
            output_tokens,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty() && !self.endpoint.is_empty()
    }
}
