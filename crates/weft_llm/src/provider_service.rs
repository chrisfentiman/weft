//! `ProviderService` trait — re-exported from `weft_llm_trait`.
//!
//! The canonical definition lives in `weft_llm_trait`. This module re-exports
//! it so the existing `use weft_llm::provider_service::ProviderService` paths
//! keep working.

pub use weft_llm_trait::ProviderService;
