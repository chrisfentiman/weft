//! Universal provider types — re-exported from `weft_llm_trait`.
//!
//! The canonical definitions live in `weft_llm_trait`. This module re-exports
//! them so the existing `use weft_llm::provider::*` paths keep working.

pub use weft_llm_trait::{
    Capability, ContentDelta, ProviderChunk, ProviderError, ProviderRequest, ProviderResponse,
    TokenUsage, ToolCallDelta, extract_text_messages,
};
