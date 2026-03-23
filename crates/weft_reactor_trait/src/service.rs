//! ServiceLocator and ChildSpawner traits.
//!
//! These traits are the abstraction boundary between the reactor's activity
//! dispatch loop and the concrete infrastructure (`Services`) and spawning
//! mechanism (`ReactorHandle`).
//!
//! `Services` in `weft_reactor` implements `ServiceLocator`.
//! `ReactorChildSpawner` in `weft_reactor` implements `ChildSpawner`.

use weft_core::ResolvedConfig;

use crate::budget::Budget;
use crate::event::PipelineEvent;
use crate::execution::{ExecutionId, RequestId, TenantId};

/// Abstracts the shared infrastructure that activities need.
///
/// Activities take `&dyn ServiceLocator` instead of the concrete `Services`
/// struct. Each method returns a trait object reference, making this trait
/// object-safe.
///
/// The concrete implementation is `Services` in `weft_reactor`, which holds
/// `Arc<dyn T>` fields for each service. The coercion from `&Services` to
/// `&dyn ServiceLocator` happens automatically at the activity call site.
///
/// All methods return references (not owned values), keeping allocation out
/// of the hot path. `memory()` returns `Option<&dyn MemoryService>` because
/// memory is optional — activities that use memory must handle the None case.
pub trait ServiceLocator: Send + Sync {
    /// Provider service: look up providers by model name, get capabilities.
    fn providers(&self) -> &dyn weft_provider_trait::ProviderService;

    /// Semantic router: score requests against routing candidates.
    fn router(&self) -> &dyn weft_router_trait::SemanticRouter;

    /// Command registry: list, describe, and execute commands.
    fn commands(&self) -> &dyn weft_commands_trait::CommandRegistry;

    /// Hook runner: execute hook chains at lifecycle points.
    fn hooks(&self) -> &dyn weft_hooks_trait::HookRunner;

    /// Resolved configuration for request processing.
    ///
    /// Returns the fully-resolved, pre-computed config snapshot. Activities
    /// access hot config (system prompt, thresholds, pre-computed provider data)
    /// via this method. For per-request consistency, prefer `ActivityInput.config`
    /// which is snapshotted once at request entry.
    fn resolved_config(&self) -> &ResolvedConfig;

    /// Memory service. `None` when memory is disabled in config.
    fn memory(&self) -> Option<&dyn weft_memory_trait::MemoryService>;
}

/// Collects the parameters for spawning a child execution.
///
/// Passed to `ChildSpawner::spawn_child` in place of the previous 8-parameter
/// signature. `parent_cancel` is kept separate because it is a borrowed
/// reference — embedding it would require a lifetime parameter on the struct.
pub struct SpawnRequest {
    pub request: weft_core::WeftRequest,
    pub tenant_id: TenantId,
    pub request_id: RequestId,
    pub parent_id: ExecutionId,
    pub parent_budget: Budget,
    pub parent_event_tx: tokio::sync::mpsc::Sender<PipelineEvent>,
    pub pipeline_name: String,
}

/// Spawns child executions from within an activity.
///
/// Only `GenerateActivity` uses this. The Reactor provides an implementation
/// wrapping `ReactorHandle`. Activities access it via `ActivityInput::child_spawner`.
///
/// The trait is object-safe: `Arc<dyn ChildSpawner>` works. With `#[async_trait]`,
/// the returned future borrows `parent_cancel` for the duration of the call.
/// `GenerateActivity` holds the token and awaits inline — no issue.
#[async_trait::async_trait]
pub trait ChildSpawner: Send + Sync {
    /// Spawn a child execution and wait for it to complete.
    ///
    /// Returns the child's final `Budget` so the parent can deduct usage via
    /// `parent_budget.deduct_child_usage`. Returns `Err(String)` on failure
    /// (depth limit, child execution error, etc.).
    ///
    /// `parent_cancel`: when Some, the child's cancellation token is linked
    /// as a child of this token, so cancelling the parent propagates.
    async fn spawn_child(
        &self,
        req: SpawnRequest,
        parent_cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<Budget, String>;
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::sync::Arc;

    // ── ServiceLocator object-safety ───────────────────────────────────────
    //
    // The spec (Section 10.3) requires: verify with `let _: &dyn ServiceLocator = ...;`

    struct MockProviders;
    impl weft_provider_trait::ProviderService for MockProviders {
        fn get(&self, _: &str) -> &Arc<dyn weft_provider_trait::Provider> {
            unimplemented!("mock")
        }
        fn model_id(&self, _: &str) -> Option<&str> {
            None
        }
        fn max_tokens_for(&self, _: &str) -> Option<u32> {
            None
        }
        fn default_provider(&self) -> &Arc<dyn weft_provider_trait::Provider> {
            unimplemented!("mock")
        }
        fn default_name(&self) -> &str {
            "mock"
        }
        fn models_with_capability(
            &self,
            _: &weft_provider_trait::Capability,
        ) -> &std::collections::HashSet<String> {
            unimplemented!("mock")
        }
        fn model_has_capability(&self, _: &str, _: &weft_provider_trait::Capability) -> bool {
            false
        }
        fn model_capabilities(
            &self,
            _: &str,
        ) -> Option<&std::collections::HashSet<weft_provider_trait::Capability>> {
            None
        }
    }

    struct MockRouter;

    #[async_trait::async_trait]
    impl weft_router_trait::SemanticRouter for MockRouter {
        async fn route(
            &self,
            _: &str,
            _: &[(
                weft_router_trait::RoutingDomainKind,
                Vec<weft_router_trait::RoutingCandidate>,
            )],
        ) -> Result<weft_router_trait::RoutingDecision, weft_router_trait::RouterError> {
            unimplemented!("mock")
        }

        async fn score_memory_candidates(
            &self,
            _: &str,
            _: &[weft_router_trait::RoutingCandidate],
        ) -> Result<Vec<weft_router_trait::ScoredCandidate>, weft_router_trait::RouterError>
        {
            unimplemented!("mock")
        }
    }

    struct MockCommands;

    #[async_trait::async_trait]
    impl weft_commands_trait::CommandRegistry for MockCommands {
        async fn list_commands(
            &self,
        ) -> Result<Vec<weft_core::CommandStub>, weft_commands_trait::CommandError> {
            Ok(vec![])
        }

        async fn describe_command(
            &self,
            _: &str,
        ) -> Result<weft_core::CommandDescription, weft_commands_trait::CommandError> {
            unimplemented!("mock")
        }

        async fn execute_command(
            &self,
            _: &weft_core::CommandInvocation,
        ) -> Result<weft_core::CommandResult, weft_commands_trait::CommandError> {
            unimplemented!("mock")
        }
    }

    struct MockHooks;

    #[async_trait::async_trait]
    impl weft_hooks_trait::HookRunner for MockHooks {
        async fn run_chain(
            &self,
            _event: weft_core::HookEvent,
            payload: serde_json::Value,
            _matcher_target: Option<&str>,
        ) -> weft_hooks_trait::HookChainResult {
            weft_hooks_trait::HookChainResult::allow(payload)
        }
    }

    struct MockMemory;

    #[async_trait::async_trait]
    impl weft_memory_trait::MemoryService for MockMemory {
        async fn recall(&self, _: &[String], _: &str, _: &str) -> weft_core::CommandResult {
            unimplemented!("mock")
        }

        async fn remember(&self, _: &[String], _: &str) -> weft_core::CommandResult {
            unimplemented!("mock")
        }

        fn is_configured(&self) -> bool {
            true
        }

        fn stores(&self) -> Vec<weft_memory_trait::StoreInfo> {
            vec![]
        }
    }

    struct MockServiceLocator {
        providers: MockProviders,
        router: MockRouter,
        commands: MockCommands,
        hooks: MockHooks,
        memory: Option<MockMemory>,
        resolved_config: weft_core::ResolvedConfig,
    }

    impl MockServiceLocator {
        fn new(with_memory: bool) -> Self {
            let config: weft_core::WeftConfig = toml::from_str(
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

            Self {
                providers: MockProviders,
                router: MockRouter,
                commands: MockCommands,
                hooks: MockHooks,
                memory: if with_memory { Some(MockMemory) } else { None },
                resolved_config: weft_core::ResolvedConfig::from_operator(&config),
            }
        }
    }

    impl ServiceLocator for MockServiceLocator {
        fn providers(&self) -> &dyn weft_provider_trait::ProviderService {
            &self.providers
        }

        fn router(&self) -> &dyn weft_router_trait::SemanticRouter {
            &self.router
        }

        fn commands(&self) -> &dyn weft_commands_trait::CommandRegistry {
            &self.commands
        }

        fn hooks(&self) -> &dyn weft_hooks_trait::HookRunner {
            &self.hooks
        }

        fn resolved_config(&self) -> &weft_core::ResolvedConfig {
            &self.resolved_config
        }

        fn memory(&self) -> Option<&dyn weft_memory_trait::MemoryService> {
            self.memory
                .as_ref()
                .map(|m| m as &dyn weft_memory_trait::MemoryService)
        }
    }

    /// Spec Section 10.3: verify ServiceLocator is object-safe with `&dyn ServiceLocator`.
    #[test]
    fn service_locator_is_object_safe() {
        let loc = MockServiceLocator::new(false);
        // This line verifies object safety: if ServiceLocator is not object-safe,
        // this would be a compile error.
        let _: &dyn ServiceLocator = &loc;
    }

    #[test]
    fn service_locator_memory_none_when_not_configured() {
        let loc = MockServiceLocator::new(false);
        let sl: &dyn ServiceLocator = &loc;
        assert!(sl.memory().is_none());
    }

    #[test]
    fn service_locator_memory_some_when_configured() {
        let loc = MockServiceLocator::new(true);
        let sl: &dyn ServiceLocator = &loc;
        assert!(sl.memory().is_some());
    }

    #[test]
    fn service_locator_default_name_accessible() {
        let loc = MockServiceLocator::new(false);
        let sl: &dyn ServiceLocator = &loc;
        assert_eq!(sl.providers().default_name(), "mock");
    }

    // ── ChildSpawner object-safety ─────────────────────────────────────────
    //
    // Spec (Section 3.5): verify `Arc<dyn ChildSpawner>` works.

    struct MockChildSpawner;

    #[async_trait::async_trait]
    impl ChildSpawner for MockChildSpawner {
        async fn spawn_child(
            &self,
            req: SpawnRequest,
            _parent_cancel: Option<&tokio_util::sync::CancellationToken>,
        ) -> Result<Budget, String> {
            // Return the parent budget unchanged — mock doesn't actually spawn.
            Ok(req.parent_budget)
        }
    }

    /// Spec Section 3.5: verify `Arc<dyn ChildSpawner>` compiles (object-safe).
    #[test]
    fn child_spawner_is_object_safe() {
        let _arc: Arc<dyn ChildSpawner> = Arc::new(MockChildSpawner);
    }

    #[tokio::test]
    async fn child_spawner_mock_returns_budget() {
        use chrono::Utc;

        let spawner = MockChildSpawner;
        let budget = Budget::new(10, 5, 3, Utc::now() + chrono::Duration::hours(1));
        let (tx, _rx) = tokio::sync::mpsc::channel(1);

        let result = spawner
            .spawn_child(
                SpawnRequest {
                    request: weft_core::WeftRequest {
                        messages: vec![],
                        routing: weft_core::ModelRoutingInstruction::parse("auto"),
                        options: weft_core::SamplingOptions::default(),
                    },
                    tenant_id: TenantId("t1".to_string()),
                    request_id: RequestId("r1".to_string()),
                    parent_id: ExecutionId::new(),
                    parent_budget: budget.clone(),
                    parent_event_tx: tx,
                    pipeline_name: "default".to_string(),
                },
                None,
            )
            .await;

        assert!(result.is_ok());
    }
}
