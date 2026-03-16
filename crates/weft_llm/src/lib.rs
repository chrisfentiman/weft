//! `weft_llm` — Universal AI provider trait and implementations.
//!
//! Contains:
//! - `Provider` trait for executing requests against an AI provider backend
//! - `ProviderError` error type
//! - `ProviderRequest` / `ProviderResponse` enums covering all capability types
//! - `TokenUsage` for provider-reported token counts
//! - `Capability` newtype with well-known constants
//! - `extract_text_messages` utility for extracting text from `WeftMessage` slices
//! - `AnthropicProvider`: Anthropic Messages API implementation
//! - `OpenAIProvider`: OpenAI Chat Completions API implementation
//! - `ProviderRegistry`: Registry of named providers keyed by model routing name

pub mod anthropic;
pub mod openai;
pub mod provider;
pub mod provider_service;
pub mod registry;
pub mod rhai_provider;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;
pub use provider::{
    Capability, ProviderError, ProviderRequest, ProviderResponse, TokenUsage, extract_text_messages,
};
pub use provider_service::ProviderService;
pub use registry::ProviderRegistry;
pub use rhai_provider::RhaiProvider;

use async_trait::async_trait;

/// A universal AI provider.
///
/// Providers handle requests they support and return `ProviderError::Unsupported`
/// for request types they cannot handle.
///
/// The request carries `Vec<WeftMessage>` (Weft Wire format). The system prompt,
/// if present, is `messages[0]` with `Role::System` -- a positional convention
/// set by the gateway during context assembly. Each provider extracts what it
/// needs from the `WeftMessage` content parts and translates to its wire format.
///
/// Send + Sync + 'static: shared across request handlers via Arc.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Execute a provider request and return the response.
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError>;

    /// The name of this provider instance, for logging and diagnostics.
    ///
    /// This is informational -- the actual capability filtering happens at the
    /// model level via config, not at the provider level. A single provider
    /// instance may serve multiple models with different capabilities.
    fn name(&self) -> &str;
}
