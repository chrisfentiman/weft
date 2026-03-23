//! `weft_providers` -- Provider implementations and wire format translation.
//!
//! Contains:
//! - `Provider` trait for executing requests against a provider backend
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

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;
pub use provider::{
    Capability, ContentDelta, ProviderChunk, ProviderError, ProviderRequest, ProviderResponse,
    TokenUsage, ToolCallDelta, extract_text_messages,
};
pub use provider_service::ProviderService;
pub use registry::ProviderRegistry;
pub use rhai_provider::RhaiProvider;
// Re-export Provider trait from weft_provider_trait so existing import paths work.
pub use weft_provider_trait::Provider;
