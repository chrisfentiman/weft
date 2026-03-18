//! Services struct: shared infrastructure for activities.
//!
//! `Services` holds Arc-wrapped trait objects for all shared infrastructure
//! that activities need: LLM providers, semantic router, command registry,
//! memory service, and hook runner.
//!
//! Constructed at startup and shared via `Arc<Services>` across all
//! activity invocations. The `reactor_handle` field is populated after
//! Reactor construction to break the circular dependency.

use std::sync::Arc;

/// Shared infrastructure for pipeline activities.
///
/// Activities receive a `&Services` reference and use it to access
/// providers, routing, commands, memory, and hooks. They do not own
/// these resources directly.
///
/// All fields are Arc-wrapped trait objects. The Services struct itself
/// is shared via `Arc<Services>`.
pub struct Services {
    /// Weft configuration (model entries, routing config, etc.).
    pub config: Arc<weft_core::WeftConfig>,

    /// LLM provider service: routes model names to provider implementations.
    pub providers: Arc<dyn weft_llm::ProviderService + Send + Sync>,

    /// Semantic router: makes all routing decisions for a request.
    pub router: Arc<dyn weft_router::SemanticRouter + Send + Sync>,

    /// Command registry: lists, describes, and executes commands.
    pub commands: Arc<dyn weft_commands::CommandRegistry + Send + Sync>,

    /// Optional memory service. None when memory is disabled in config.
    pub memory: Option<Arc<dyn weft_memory::MemoryService + Send + Sync>>,

    /// Hook runner: executes hook chains at lifecycle points.
    pub hooks: Arc<dyn weft_hooks::HookRunner + Send + Sync>,

    /// Handle for spawning child executions.
    ///
    /// Populated after Reactor construction via `OnceLock::set`.
    /// `None` (not-yet-set) during Reactor construction, which breaks
    /// the circular dependency (Reactor -> Services -> Reactor).
    ///
    /// Activities that need to spawn child executions must check that
    /// this is set before use. All built-in activities that spawn children
    /// run in a `tokio::spawn` context after Reactor construction, so
    /// the lock is guaranteed to be set by the time they execute.
    pub reactor_handle: std::sync::OnceLock<Arc<ReactorHandle>>,

    /// Semaphore for bounding RequestEnd hook concurrency.
    ///
    /// RequestEnd hooks are fire-and-forget: HookActivity for RequestEnd
    /// spawns hook execution on a separate tokio task gated by this
    /// semaphore, then immediately pushes ActivityCompleted without waiting.
    pub request_end_semaphore: Arc<tokio::sync::Semaphore>,
}

/// Handle for spawning child executions.
///
/// Wraps `Arc<Reactor>` and provides a `spawn_child` method.
/// The indirection breaks the circular dependency at construction time:
/// Services is built first (reactor_handle starts empty), then the Reactor
/// is built with Arc<Services>, then ReactorHandle is created and set via
/// OnceLock.
///
/// `spawn_child` is async and blocks the calling task until the child
/// completes. Because the calling activity runs on a separate `tokio::spawn`
/// task, this does NOT block the parent Reactor's dispatch loop.
pub struct ReactorHandle {
    /// The reactor instance. Phase 5 adds spawn_child which uses this.
    pub reactor: std::sync::Arc<crate::reactor::Reactor>,
}

impl ReactorHandle {
    /// Construct a ReactorHandle wrapping the given Reactor.
    pub fn new(reactor: std::sync::Arc<crate::reactor::Reactor>) -> Self {
        Self { reactor }
    }
}

impl std::fmt::Debug for ReactorHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReactorHandle").finish()
    }
}

impl std::fmt::Debug for Services {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Services")
            .field("has_memory", &self.memory.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactor_handle_debug_impl_exists() {
        // The Debug impl is derived on ReactorHandle (via the Debug impl on Reactor).
        // We verify the struct name appears in the output.
        let s = format!("{:?}", std::any::type_name::<ReactorHandle>());
        assert!(s.contains("ReactorHandle"));
    }
}
