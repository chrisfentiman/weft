//! `weft_llm` — LLM provider trait and implementations.
//!
//! Contains:
//! - `LlmProvider` trait for sending conversations to an LLM backend
//! - `LlmError` error type
//! - `CompletionOptions` for per-request options
//! - `CompletionResponse` and `LlmUsage` for provider responses
//! - `AnthropicProvider`: Anthropic Messages API implementation
//! - `OpenAIProvider`: OpenAI Chat Completions API implementation
//! - `ProviderRegistry`: Registry of named LLM providers keyed by model routing name

pub mod anthropic;
pub mod openai;
pub mod registry;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;
pub use registry::ProviderRegistry;

use async_trait::async_trait;
use weft_core::Message;

/// Options forwarded from the client request to the LLM provider.
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    /// Maximum tokens in response. If None, use provider/config default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature. If None, use provider default.
    pub temperature: Option<f32>,
    /// Model identifier to use for this request.
    ///
    /// **Contract:** The engine guarantees this is always `Some` when calling providers.
    /// The engine resolves the model from the routing decision and `ProviderRegistry::model_id()`
    /// before constructing `CompletionOptions`. The `Option` exists only because
    /// `CompletionOptions` derives `Default` (used in tests); production code paths
    /// always set this field.
    ///
    /// **Provider behavior:** Providers MUST return `LlmError` if `model` is `None`.
    /// This is a defensive check -- it should never fire if the engine is wired correctly.
    pub model: Option<String>,
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
