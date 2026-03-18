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
/// Wraps `Arc<Reactor>` and provides a `spawn_child` method that
/// child-capable activities can call. The indirection is needed to
/// break the circular dependency at construction time.
///
/// `spawn_child` is async and blocks the calling task until the child
/// completes. Because the calling activity is itself running on a
/// separate `tokio::spawn` task, this does NOT block the parent
/// Reactor's dispatch loop.
pub struct ReactorHandle {
    /// The reactor instance. Defined as a type-erased value to avoid
    /// introducing a circular type dependency here. Phase 4 will replace
    /// this with a concrete field once Reactor is implemented.
    ///
    /// Stored as a `Box<dyn std::any::Any + Send + Sync>` to avoid
    /// importing `Reactor` here (Reactor imports Services, which would
    /// create a cycle if Services imported Reactor).
    inner: Box<dyn std::any::Any + Send + Sync>,
}

impl ReactorHandle {
    /// Construct a ReactorHandle from any value.
    ///
    /// Phase 4 will replace this with a typed constructor taking `Arc<Reactor>`.
    pub fn new(inner: impl std::any::Any + Send + Sync + 'static) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Downcast the inner value to the given type.
    ///
    /// Used by Phase 4's Reactor to recover the `Arc<Reactor>` from the handle.
    pub fn downcast_ref<T: std::any::Any>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
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
    fn reactor_handle_debug() {
        let handle = ReactorHandle::new(42u32);
        let s = format!("{handle:?}");
        assert!(s.contains("ReactorHandle"));
    }

    #[test]
    fn reactor_handle_downcast() {
        let handle = ReactorHandle::new(99u64);
        let val = handle.downcast_ref::<u64>();
        assert_eq!(val, Some(&99u64));
    }

    #[test]
    fn reactor_handle_downcast_wrong_type_is_none() {
        let handle = ReactorHandle::new(99u64);
        let val = handle.downcast_ref::<u32>();
        assert!(val.is_none());
    }
}
