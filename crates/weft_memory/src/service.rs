//! Re-exports `MemoryService` and `StoreInfo` from `weft_memory_trait`.
//!
//! The canonical definitions live in `weft_memory_trait`. This module re-exports
//! them so the existing `use weft_memory::service::MemoryService` paths keep working.

pub use weft_memory_trait::{MemoryService, StoreInfo};
