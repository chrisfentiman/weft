//! `ProviderService` trait — re-exported from `weft_provider_trait`.
//!
//! The canonical definition lives in `weft_provider_trait`. This module re-exports
//! it so the existing `use weft_providers::provider_service::ProviderService` paths
//! keep working.

pub use weft_provider_trait::ProviderService;
