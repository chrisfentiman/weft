//! `MemoryService` trait and `StoreInfo` domain type.
//!
//! `MemoryService` is a pure domain abstraction over one or more named memory stores.
//! It knows how to recall and remember — it does NOT know about routing candidates,
//! scoring thresholds, or validation. Those concerns belong to the engine/router layer.
//!
//! The engine builds `RoutingCandidate`s from `stores()`, asks the router to score them,
//! then calls `recall` or `remember` with the resolved `store_ids`. Memory never imports
//! routing types, preventing a circular dependency with `weft_router`.

use weft_core::CommandResult;

/// Information about a single configured memory store.
///
/// Returned by `MemoryService::stores()` so the engine can build routing candidates
/// without importing routing types into this crate.
#[derive(Debug, Clone)]
pub struct StoreInfo {
    /// The store's unique name (from config).
    pub name: String,
    /// Capability labels, e.g. `["read", "write"]` or `["read"]`.
    pub capabilities: Vec<String>,
    /// Example phrases used by the semantic router for candidate scoring.
    pub examples: Vec<String>,
}

/// Pure domain abstraction over memory store operations.
///
/// Implementations own whatever backing store(s) they need. The engine
/// calls `stores()` at startup to build routing candidates; at request time
/// it calls `recall` or `remember` with the routing-resolved store IDs.
///
/// ## Design invariants
///
/// - `recall` and `remember` accept empty `store_ids` — the implementation
///   falls back to a sensible default (all readable stores for recall,
///   first writable store for remember).
/// - Neither method panics. Partial failures are folded into the `CommandResult`.
/// - `is_configured()` returning `false` signals the engine to skip memory
///   routing entirely for this request.
#[async_trait::async_trait]
pub trait MemoryService: Send + Sync + 'static {
    /// Execute a recall (query) across one or more named stores.
    ///
    /// `store_ids` — which stores to query; empty means "all readable stores".
    /// `query` — the explicit query string (from the `/recall` argument if provided).
    /// `user_message` — fallback query when no explicit `query` argument is given.
    async fn recall(&self, store_ids: &[String], query: &str, user_message: &str) -> CommandResult;

    /// Execute a remember (store) operation across one or more named stores.
    ///
    /// `store_ids` — which stores to write; empty means "first writable store".
    /// `content` — the text to store.
    async fn remember(&self, store_ids: &[String], content: &str) -> CommandResult;

    /// Whether any stores are configured.
    ///
    /// The engine uses this to gate memory routing — if `false`, all memory
    /// commands return an unconfigured error immediately.
    fn is_configured(&self) -> bool;

    /// Return metadata for all configured stores.
    ///
    /// The engine converts these into `RoutingCandidate`s for the semantic router.
    /// Called once at startup; the returned vec is stable for the lifetime of the service.
    fn stores(&self) -> Vec<StoreInfo>;
}
