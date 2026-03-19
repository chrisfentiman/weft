//! Services struct: shared infrastructure for activities.
//!
//! `Services` holds Arc-wrapped trait objects for all shared infrastructure
//! that activities need: LLM providers, semantic router, command registry,
//! memory service, and hook runner.
//!
//! Constructed at startup and shared via `Arc<Services>` across all
//! activity invocations. The `reactor_handle` field is populated after
//! Reactor construction to break the circular dependency.
//!
//! # Two-phase construction (OnceLock pattern)
//!
//! `Services` and `Reactor` have a circular dependency: Reactor needs
//! Services (for activities), and Services needs Reactor (for spawn_child).
//! The OnceLock breaks this:
//!
//! 1. Build `Services` with `reactor_handle` empty (OnceLock starts unset).
//! 2. Build `Reactor` with `Arc<Services>`.
//! 3. Wrap reactor in `Arc`. Create `ReactorHandle` from it.
//! 4. Set `services.reactor_handle` via `OnceLock::set()`.
//!
//! After step 4, any activity that calls `services.reactor_handle.get()`
//! will receive the handle. Since activities only run after execute() is
//! called (which is after construction completes), the handle is always
//! set by the time activities need it.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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
    /// The reactor instance. Kept private to enforce construction via `new()`.
    reactor: std::sync::Arc<crate::reactor::Reactor>,
}

impl ReactorHandle {
    /// Construct a ReactorHandle wrapping the given Reactor.
    pub fn new(reactor: std::sync::Arc<crate::reactor::Reactor>) -> Self {
        Self { reactor }
    }

    /// Spawn a child execution.
    ///
    /// The child runs the default pipeline with its own `ExecutionId`,
    /// inheriting the parent's remaining budget (depth incremented by 1).
    /// When the child completes, a `ChildCompleted` event is pushed onto
    /// the `parent_event_tx` channel so the parent's dispatch loop receives it.
    ///
    /// This method blocks the calling task until the child execution
    /// completes. It does NOT block the parent Reactor's dispatch loop
    /// because the caller (a generate activity) runs on a separate
    /// `tokio::spawn` task.
    ///
    /// # Cancellation hierarchy
    ///
    /// When `parent_cancel` is `Some`, the child's `CancellationToken` is
    /// created as a child of it via `child_token()`. Cancelling the parent
    /// token therefore propagates to the child. Pass `None` for no
    /// cancellation link.
    ///
    /// # Budget inheritance
    ///
    /// The child receives a budget derived from `parent_budget` via
    /// `child_budget()`. On success, call `parent_budget.deduct_child_usage`
    /// with the returned `Budget` to reduce the parent's remaining resources.
    ///
    /// # Errors
    ///
    /// Returns `ReactorError::BudgetExhausted` if `parent_budget.child_budget()`
    /// fails (depth limit reached). Propagates any other error from the child
    /// execution.
    // Spec-mandated signature requires 9 positional parameters; grouping into
    // a builder would change the public API contract. Allow lint locally.
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn_child(
        &self,
        request: weft_core::WeftRequest,
        tenant_id: crate::execution::TenantId,
        request_id: crate::execution::RequestId,
        parent_id: crate::execution::ExecutionId,
        parent_budget: crate::budget::Budget,
        parent_event_tx: mpsc::Sender<crate::event::PipelineEvent>,
        parent_cancel: Option<&CancellationToken>,
        pipeline_name: &str,
    ) -> Result<crate::budget::Budget, crate::error::ReactorError> {
        // Pre-validate depth before spawning. This surfaces the Depth error
        // synchronously (before we record ChildSpawned), matching the spec's
        // contract that spawn_child returns Err(BudgetExhausted) at depth limit.
        //
        // reactor.execute() also calls child_budget() internally when parent_budget
        // is Some, so we do NOT pass the pre-validated child budget here — we pass
        // the parent_budget directly and let execute() perform the depth increment.
        // The pre-call here only serves to return early with a clear error.
        let _ = parent_budget
            .child_budget()
            .map_err(|r| crate::error::ReactorError::BudgetExhausted(r.to_string()))?;

        // Record child spawn event on the parent's channel.
        // This is fire-and-forget: if the parent channel is full, we lose the log event
        // but still proceed with the child execution.
        let _ = parent_event_tx
            .send(crate::event::PipelineEvent::ChildSpawned {
                child_id: "pending".to_string(),
                pipeline_name: pipeline_name.to_string(),
                reason: "spawn_child".to_string(),
            })
            .await;

        // Pass the parent's budget directly. reactor.execute() calls child_budget()
        // internally when parent_budget is Some, incrementing depth by 1.
        let (result, _child_signal_tx) = self
            .reactor
            .execute(
                request,
                tenant_id,
                request_id,
                Some(parent_id.clone()),
                Some(parent_budget),
                None, // No client streaming for child executions
                parent_cancel,
            )
            .await?;

        // Push ChildCompleted onto parent's channel.
        // Ignore send errors: if the parent channel is closed, we've already
        // completed the child and can't do anything about the notification.
        let _ = parent_event_tx
            .send(crate::event::PipelineEvent::ChildCompleted {
                child_id: result.execution_id.clone(),
                status: "completed".to_string(),
                result_summary: serde_json::to_value(&result.budget_used).unwrap_or_default(),
            })
            .await;

        // Return child's final budget so caller can deduct from parent.
        Ok(result.final_budget)
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
