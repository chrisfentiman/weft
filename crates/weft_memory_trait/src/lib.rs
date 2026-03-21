//! `weft_memory_trait` — Memory service trait and domain types.
//!
//! Contains the `MemoryService` trait and `StoreInfo`. The implementations live
//! in `weft_memory`, which depends on this crate.
//!
//! Consumers that need the trait boundary without the implementation (e.g.
//! `weft_reactor_trait`) depend on this crate directly.

use weft_core::CommandResult;

// ── StoreInfo ──────────────────────────────────────────────────────────────

/// Information about a single configured memory store.
///
/// Returned by `MemoryService::stores()` so the engine can build routing candidates
/// without importing routing types into this crate.
#[derive(Debug, Clone)]
pub struct StoreInfo {
    pub name: String,
    pub capabilities: Vec<String>,
    pub examples: Vec<String>,
}

// ── MemoryService trait ────────────────────────────────────────────────────

/// Pure domain abstraction over memory store operations.
///
/// Implementations own whatever backing store(s) they need. The engine
/// calls `stores()` at startup to build routing candidates; at request time
/// it calls `recall` or `remember` with the routing-resolved store IDs.
///
/// `Send + Sync + 'static`: shared via `Arc` across async request handlers.
#[async_trait::async_trait]
pub trait MemoryService: Send + Sync + 'static {
    /// Execute a recall (query) across one or more named stores.
    async fn recall(&self, store_ids: &[String], query: &str, user_message: &str) -> CommandResult;

    /// Execute a remember (store) operation across one or more named stores.
    async fn remember(&self, store_ids: &[String], content: &str) -> CommandResult;

    /// Whether any stores are configured.
    fn is_configured(&self) -> bool;

    /// Return metadata for all configured stores.
    fn stores(&self) -> Vec<StoreInfo>;
}
