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
    Capability, ContentDelta, ProviderChunk, ProviderError, ProviderRequest, ProviderResponse,
    TokenUsage, ToolCallDelta, extract_text_messages,
};
pub use provider_service::ProviderService;
pub use registry::ProviderRegistry;
pub use rhai_provider::RhaiProvider;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

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

    /// Execute a request and return a stream of response chunks.
    ///
    /// Each chunk is a partial response (`ProviderChunk::Delta`) or the complete
    /// response (`ProviderChunk::Complete`). The stream ends after the `Complete`
    /// chunk is yielded.
    ///
    /// Providers that support true token-level streaming override this method to
    /// yield `Delta` chunks as tokens arrive from the API. Providers that do not
    /// support streaming use the default implementation, which calls `execute()`
    /// and wraps the result in a single-element `Complete` stream so that
    /// `GenerateActivity` can use a unified code path for all providers.
    ///
    /// The return type uses `Pin<Box<dyn Stream>>` to enable object-safe dispatch
    /// through `dyn Provider`.
    async fn execute_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        let response = self.execute(request).await?;
        Ok(Box::pin(futures::stream::once(async move {
            Ok(ProviderChunk::Complete(response))
        })))
    }

    /// The name of this provider instance, for logging and diagnostics.
    ///
    /// This is informational -- the actual capability filtering happens at the
    /// model level via config, not at the provider level. A single provider
    /// instance may serve multiple models with different capabilities.
    fn name(&self) -> &str;
}
