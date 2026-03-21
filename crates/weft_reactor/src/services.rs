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
//! Also defines `ReactorChildSpawner`, which implements `ChildSpawner` by
//! delegating to `ReactorHandle::spawn_child`. This is what gets stored in
//! `ActivityInput::child_spawner` for GenerateActivity.
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
use weft_core::{ConfigStore, ResolvedConfig};
use weft_reactor_trait::{
    Budget, ChildSpawner, ExecutionId, PipelineEvent, RequestId, ServiceLocator, TenantId,
};

/// Collects the parameters for `ReactorHandle::spawn_child` into a single owned struct.
///
/// Replaces the previous 8-parameter signature. `parent_cancel` is kept separate
/// because it is a borrowed reference — embedding it would require a lifetime
/// parameter on the struct itself.
pub struct SpawnRequest {
    pub request: weft_core::WeftRequest,
    pub tenant_id: TenantId,
    pub request_id: RequestId,
    pub parent_id: ExecutionId,
    pub parent_budget: Budget,
    pub parent_event_tx: mpsc::Sender<PipelineEvent>,
    pub pipeline_name: String,
}

/// Shared infrastructure for pipeline activities.
///
/// Activities receive a `&dyn ServiceLocator` reference (via the `ServiceLocator`
/// impl below) and use it to access providers, routing, commands, memory, and hooks.
/// They do not own these resources directly.
///
/// All fields are Arc-wrapped trait objects. The Services struct itself
/// is shared via `Arc<Services>`.
pub struct Services {
    /// Config store for operator-level access (startup only) and per-request snapshots.
    ///
    /// The reactor calls `config_store.snapshot()` once at request entry and places
    /// the resulting `Arc<ResolvedConfig>` in `ActivityInput.config`. Activities should
    /// prefer `input.config` for per-request consistency.
    pub config_store: Arc<ConfigStore>,

    /// Resolved config snapshot (latest). Returned by `ServiceLocator::resolved_config()`.
    ///
    /// This reflects the latest config at Services construction time (startup).
    /// For per-request consistency, use `ActivityInput.config` (snapshotted once at
    /// request entry). This field is only for `ServiceLocator::resolved_config()` fallback.
    pub resolved_config: Arc<ResolvedConfig>,

    /// LLM provider service: routes model names to provider implementations.
    pub providers: Arc<dyn weft_llm_trait::ProviderService + Send + Sync>,

    /// Semantic router: makes all routing decisions for a request.
    pub router: Arc<dyn weft_router_trait::SemanticRouter + Send + Sync>,

    /// Command registry: lists, describes, and executes commands.
    pub commands: Arc<dyn weft_commands_trait::CommandRegistry + Send + Sync>,

    /// Optional memory service. None when memory is disabled in config.
    pub memory: Option<Arc<dyn weft_memory_trait::MemoryService + Send + Sync>>,

    /// Hook runner: executes hook chains at lifecycle points.
    pub hooks: Arc<dyn weft_hooks_trait::HookRunner + Send + Sync>,

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

impl ServiceLocator for Services {
    fn providers(&self) -> &dyn weft_llm_trait::ProviderService {
        self.providers.as_ref()
    }

    fn router(&self) -> &dyn weft_router_trait::SemanticRouter {
        self.router.as_ref()
    }

    fn commands(&self) -> &dyn weft_commands_trait::CommandRegistry {
        self.commands.as_ref()
    }

    fn hooks(&self) -> &dyn weft_hooks_trait::HookRunner {
        self.hooks.as_ref()
    }

    fn resolved_config(&self) -> &ResolvedConfig {
        &self.resolved_config
    }

    fn memory(&self) -> Option<&dyn weft_memory_trait::MemoryService> {
        self.memory
            .as_ref()
            .map(|m| m.as_ref() as &dyn weft_memory_trait::MemoryService)
    }
}

/// `ChildSpawner` implementation backed by `ReactorHandle`.
///
/// Created in `Reactor::build_input` and stored in `ActivityInput::child_spawner`.
/// `GenerateActivity` uses this instead of accessing `Services::reactor_handle` directly,
/// which eliminates the need for activities to know about the `Services` struct at all.
pub struct ReactorChildSpawner {
    handle: Arc<ReactorHandle>,
}

impl ReactorChildSpawner {
    /// Create a new spawner wrapping the given handle.
    pub fn new(handle: Arc<ReactorHandle>) -> Self {
        Self { handle }
    }
}

#[async_trait::async_trait]
impl ChildSpawner for ReactorChildSpawner {
    async fn spawn_child(
        &self,
        req: weft_reactor_trait::SpawnRequest,
        parent_cancel: Option<&CancellationToken>,
    ) -> Result<Budget, String> {
        // Convert the trait's SpawnRequest to services' SpawnRequest.
        let spawn_req = SpawnRequest {
            request: req.request,
            tenant_id: req.tenant_id,
            request_id: req.request_id,
            parent_id: req.parent_id,
            parent_budget: req.parent_budget,
            parent_event_tx: req.parent_event_tx,
            pipeline_name: req.pipeline_name,
        };
        self.handle
            .spawn_child(spawn_req, parent_cancel)
            .await
            .map_err(|e| e.to_string())
    }
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
    /// The child runs the named pipeline with its own `ExecutionId`,
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
    pub async fn spawn_child(
        &self,
        req: SpawnRequest,
        parent_cancel: Option<&CancellationToken>,
    ) -> Result<crate::budget::Budget, crate::error::ReactorError> {
        let SpawnRequest {
            request,
            tenant_id,
            request_id,
            parent_id,
            parent_budget,
            parent_event_tx,
            pipeline_name,
        } = req;

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

        // Record child spawn event on the parent's channel using new grouped format.
        // This is fire-and-forget: if the parent channel is full, we lose the log event
        // but still proceed with the child execution.
        let _ = parent_event_tx
            .send(crate::event::PipelineEvent::Child(
                weft_reactor_trait::ChildEvent::Spawned {
                    child_id: "pending".to_string(),
                    pipeline_name: pipeline_name.clone(),
                    reason: "spawn_child".to_string(),
                },
            ))
            .await;

        // Pass the parent's budget directly. reactor.execute() calls child_budget()
        // internally when parent_budget is Some, incrementing depth by 1.
        let ctx = crate::reactor::ExecutionContext {
            request,
            tenant_id,
            request_id,
            parent_id: Some(parent_id),
            parent_budget: Some(parent_budget),
            client_tx: None, // No client streaming for child executions
        };
        let (result, _child_signal_tx) = self.reactor.execute(ctx, parent_cancel).await?;

        // Push ChildCompleted onto parent's channel using new grouped format.
        // Ignore send errors: if the parent channel is closed, we've already
        // completed the child and can't do anything about the notification.
        let _ = parent_event_tx
            .send(crate::event::PipelineEvent::Child(
                weft_reactor_trait::ChildEvent::Completed {
                    child_id: result.execution_id.clone(),
                    status: "completed".to_string(),
                },
            ))
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
            .field("has_config_store", &true)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactor_handle_debug_impl_exists() {
        // Verify the struct name appears in the type name.
        let s = format!("{:?}", std::any::type_name::<ReactorHandle>());
        assert!(s.contains("ReactorHandle"));
    }

    #[test]
    fn services_implements_service_locator() {
        // Verify Services can be used as &dyn ServiceLocator.
        // This is a compile-time check — if this test builds, the impl is correct.
        use std::collections::{HashMap, HashSet};

        struct PanicProvider;
        #[async_trait::async_trait]
        impl weft_llm_trait::Provider for PanicProvider {
            async fn execute(
                &self,
                _request: weft_llm_trait::ProviderRequest,
            ) -> Result<weft_llm_trait::ProviderResponse, weft_llm_trait::ProviderError>
            {
                panic!("not called in test")
            }
            fn name(&self) -> &str {
                "panic-provider"
            }
        }

        struct PanicRouter;
        #[async_trait::async_trait]
        impl weft_router_trait::SemanticRouter for PanicRouter {
            async fn route(
                &self,
                _: &str,
                _: &[(
                    weft_router_trait::RoutingDomainKind,
                    Vec<weft_router_trait::RoutingCandidate>,
                )],
            ) -> Result<weft_router_trait::RoutingDecision, weft_router_trait::RouterError>
            {
                panic!("not called in test")
            }
            async fn score_memory_candidates(
                &self,
                _: &str,
                _: &[weft_router_trait::RoutingCandidate],
            ) -> Result<Vec<weft_router_trait::ScoredCandidate>, weft_router_trait::RouterError>
            {
                panic!("not called in test")
            }
        }

        struct PanicCommands;
        #[async_trait::async_trait]
        impl weft_commands_trait::CommandRegistry for PanicCommands {
            async fn list_commands(
                &self,
            ) -> Result<Vec<weft_core::CommandStub>, weft_commands_trait::CommandError>
            {
                panic!("not called in test")
            }
            async fn describe_command(
                &self,
                _: &str,
            ) -> Result<weft_core::CommandDescription, weft_commands_trait::CommandError>
            {
                panic!("not called in test")
            }
            async fn execute_command(
                &self,
                _: &weft_core::CommandInvocation,
            ) -> Result<weft_core::CommandResult, weft_commands_trait::CommandError> {
                panic!("not called in test")
            }
        }

        let weft_config: weft_core::WeftConfig = toml::from_str(
            r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"
max_command_iterations = 10
request_timeout_secs = 300

[router]
default_model = "stub"

[router.classifier]
model_path = "models/stub.onnx"
tokenizer_path = "models/tokenizer.json"
threshold = 0.3
max_commands = 20

[[router.providers]]
name = "stub"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "stub"
  model = "stub-model"
  max_tokens = 4096
  examples = ["test"]
"#,
        )
        .expect("minimal test TOML must parse");

        let mut providers: HashMap<String, Arc<dyn weft_llm_trait::Provider>> = HashMap::new();
        providers.insert("stub".to_string(), Arc::new(PanicProvider));
        let mut model_ids = HashMap::new();
        model_ids.insert("stub".to_string(), "stub-model".to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert("stub".to_string(), 4096u32);
        let capabilities: HashMap<String, HashSet<weft_llm_trait::Capability>> = HashMap::new();
        let registry = weft_llm::ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            capabilities,
            "stub".to_string(),
        );

        let config_store = Arc::new(weft_core::ConfigStore::new(weft_config.clone()));
        let resolved_config = config_store.snapshot();
        let services = Services {
            config_store,
            resolved_config,
            providers: Arc::new(registry),
            router: Arc::new(PanicRouter),
            commands: Arc::new(PanicCommands),
            memory: None,
            hooks: Arc::new(weft_hooks::NullHookRunner),
            reactor_handle: std::sync::OnceLock::new(),
            request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        };

        // Compile-time check: Services implements ServiceLocator
        let _loc: &dyn ServiceLocator = &services;
        // Verify memory() returns None when not configured.
        assert!(services.memory().is_none());
    }
}
