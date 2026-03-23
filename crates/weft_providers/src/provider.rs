//! Universal provider types — re-exported from `weft_provider_trait`.
//!
//! The canonical definitions live in `weft_provider_trait`. This module re-exports
//! them so the existing `use weft_providers::provider::*` paths keep working.

pub use weft_provider_trait::{
    Capability, ContentDelta, ProviderChunk, ProviderError, ProviderRequest, ProviderResponse,
    TokenUsage, ToolCallDelta, extract_text_messages,
};
