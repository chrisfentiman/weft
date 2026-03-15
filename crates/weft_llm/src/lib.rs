//! `weft_llm` — LLM provider trait and implementations.
//!
//! Phase 1 stub. Full implementation in Phase 2.
//!
//! Contains:
//! - `LlmProvider` trait for sending conversations to an LLM backend
//! - `LlmError` error type
//! - `CompletionOptions` for per-request options
//! - `CompletionResponse` and `LlmUsage` for provider responses
//! - Anthropic and OpenAI provider implementations (Phase 2)

use async_trait::async_trait;
use weft_core::Message;

/// Options forwarded from the client request to the LLM provider.
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    /// Maximum tokens in response. If None, use provider/config default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature. If None, use provider default.
    pub temperature: Option<f32>,
}

/// A backend LLM provider (Anthropic, OpenAI, etc.)
///
/// Send + Sync required: shared across request handlers via Arc.
/// 'static required: no borrowed references from the provider.
#[async_trait]
pub trait LlmProvider: Send + Sync + 'static {
    /// Send a conversation to the LLM and receive the complete response.
    ///
    /// `system_prompt`: The assembled system prompt (Weft foundational + agent).
    /// `messages`: The full conversation including injected command results.
    /// `options`: Per-request options (max_tokens, temperature) forwarded from the client.
    ///
    /// Returns the assistant's response text and usage info. Not streaming -- full response.
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> Result<CompletionResponse, LlmError>;
}

/// Response from an LLM provider.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The assistant's response text.
    pub text: String,
    /// Token usage from the provider, if available.
    pub usage: Option<LlmUsage>,
}

#[derive(Debug, Clone)]
pub struct LlmUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("provider request failed: {0}")]
    RequestFailed(String),
    #[error("provider returned non-200: status={status}, body={body}")]
    ProviderError { status: u16, body: String },
    #[error("response deserialization failed: {0}")]
    DeserializationError(String),
    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
}
