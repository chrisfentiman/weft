//! Test support utilities for weft_activities.
//!
//! Provides `MockServiceLocator` (implementing `ServiceLocator` via trait objects),
//! builder functions for common test scenarios, `NullEventLog`, `make_test_input`,
//! and `collect_events`. All items are available when the `test-support` feature is
//! enabled or when running tests.

use std::sync::Arc;

use tokio::sync::mpsc;
use weft_core::{
    ContentPart, HookEvent, ModelRoutingInstruction, Role, RoutingActivity, SamplingOptions,
    Source, WeftMessage, WeftRequest,
};
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::{Provider, ProviderError};
use weft_reactor_trait::{
    ActivityInput, Budget, EventLog, EventLogError, Execution, ExecutionId, ExecutionStatus,
    PipelineEvent, RoutingSnapshot, ServiceLocator,
};

// Re-export stub types from trait crates.
pub use weft_commands::test_support::{StubCommandRegistry, reconstruct_command_error};
pub use weft_llm::test_support::{
    ChunkStreamProvider, MidStreamErrorProvider, SingleUseErrorProvider, SlowProvider,
    StubProvider, StubProviderService,
};
pub use weft_router::test_support::{ErrorRouter, StubRouter};

// ── NullEventLog ─────────────────────────────────────────────────────────────

/// A no-op [`EventLog`] implementation for activity tests.
///
/// All operations succeed and return empty/default values.
pub struct NullEventLog;

#[async_trait::async_trait]
impl EventLog for NullEventLog {
    async fn create_execution(&self, _execution: &Execution) -> Result<(), EventLogError> {
        Ok(())
    }

    async fn update_execution_status(
        &self,
        _execution_id: &ExecutionId,
        _status: ExecutionStatus,
    ) -> Result<(), EventLogError> {
        Ok(())
    }

    async fn append(
        &self,
        _execution_id: &ExecutionId,
        _event_type: &str,
        _payload: serde_json::Value,
        _schema_version: u32,
        _idempotency_key: Option<&str>,
    ) -> Result<u64, EventLogError> {
        Ok(0)
    }

    async fn read(
        &self,
        _execution_id: &ExecutionId,
        _after_sequence: Option<u64>,
    ) -> Result<Vec<weft_reactor_trait::Event>, EventLogError> {
        Ok(vec![])
    }

    async fn latest_of_type(
        &self,
        _execution_id: &ExecutionId,
        _event_type: &str,
    ) -> Result<Option<weft_reactor_trait::Event>, EventLogError> {
        Ok(None)
    }

    async fn count_by_type(
        &self,
        _execution_id: &ExecutionId,
        _event_type: &str,
    ) -> Result<u64, EventLogError> {
        Ok(0)
    }
}

// ── BlockingHookRunner ────────────────────────────────────────────────────────

/// A hook runner that blocks a specific event with a fixed reason.
///
/// All other events return `Allowed`.
struct BlockingHookRunner {
    blocking_event: HookEvent,
    reason: String,
}

#[async_trait::async_trait]
impl HookRunner for BlockingHookRunner {
    async fn run_chain(
        &self,
        event: HookEvent,
        payload: serde_json::Value,
        _matcher_target: Option<&str>,
    ) -> HookChainResult {
        if event == self.blocking_event {
            HookChainResult::Blocked {
                reason: self.reason.clone(),
                hook_name: "test-blocking-hook".to_string(),
            }
        } else {
            HookChainResult::Allowed {
                payload,
                context: None,
            }
        }
    }
}

// ── MockServiceLocator ────────────────────────────────────────────────────────

/// A `ServiceLocator` implementation backed by Arc-wrapped trait objects.
///
/// Used in `weft_activities` tests to provide a fully typed service locator
/// without depending on `weft_reactor::services::Services`.
pub struct MockServiceLocator {
    pub config: Arc<weft_core::WeftConfig>,
    pub providers: Arc<dyn weft_llm_trait::ProviderService + Send + Sync>,
    pub router: Arc<dyn weft_router_trait::SemanticRouter + Send + Sync>,
    pub commands: Arc<dyn weft_commands_trait::CommandRegistry + Send + Sync>,
    pub hooks: Arc<dyn weft_hooks_trait::HookRunner + Send + Sync>,
    pub memory: Option<Arc<dyn weft_memory_trait::MemoryService + Send + Sync>>,
    /// Semaphore for bounding RequestEnd hook concurrency in HookActivity tests.
    pub request_end_semaphore: Arc<tokio::sync::Semaphore>,
}

impl ServiceLocator for MockServiceLocator {
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

    fn config(&self) -> &Arc<weft_core::WeftConfig> {
        &self.config
    }

    fn memory(&self) -> Option<&dyn weft_memory_trait::MemoryService> {
        self.memory
            .as_ref()
            .map(|m| m.as_ref() as &dyn weft_memory_trait::MemoryService)
    }
}

// ── Config builders ───────────────────────────────────────────────────────────

/// Build minimal test `WeftConfig`.
fn make_test_config() -> weft_core::WeftConfig {
    let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are helpful."

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "stub-model"
  model = "stub-model-v1"
  examples = ["example query"]
"#;
    toml::from_str(toml).expect("test config should parse")
}

/// Build minimal test `WeftConfig` with memory configured.
fn make_test_config_with_memory() -> weft_core::WeftConfig {
    let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are helpful."

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "stub-model"
  model = "stub-model-v1"
  examples = ["example query"]

[[memory.stores]]
name = "conversations"
endpoint = "http://localhost:50052"
capabilities = ["read", "write"]
examples = ["conversation history"]
"#;
    toml::from_str(toml).expect("test config with memory should parse")
}

// ── Services builders ─────────────────────────────────────────────────────────

/// Build `MockServiceLocator` using the given provider and hook runner.
fn build_mock(
    provider: Arc<dyn Provider>,
    hooks: Arc<dyn HookRunner + Send + Sync>,
    failing_command: Option<(String, weft_commands::CommandError)>,
    failed_result_command: Option<(String, String)>,
    fail_list_commands: bool,
) -> MockServiceLocator {
    let config = Arc::new(make_test_config());
    let providers: Arc<dyn weft_llm_trait::ProviderService + Send + Sync> =
        Arc::new(StubProviderService::new(provider));
    let router: Arc<dyn weft_router_trait::SemanticRouter + Send + Sync> = Arc::new(StubRouter);
    let commands: Arc<dyn weft_commands_trait::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry {
            failing_command,
            failed_result_command,
            fail_list_commands,
        });

    MockServiceLocator {
        config,
        providers,
        router,
        commands,
        hooks,
        memory: None,
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    }
}

/// Build test `MockServiceLocator` with all null/stub implementations.
///
/// The provider returns "stub response" text. The hook runner allows all events.
/// The command registry returns a single stub command.
pub fn make_test_services() -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` whose provider returns a fixed text response.
pub fn make_test_services_with_response(response_text: &str) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new(response_text));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` whose provider returns a `ProviderError`.
pub fn make_test_services_with_error(err: ProviderError) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(SingleUseErrorProvider::new(err));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` where executing a specific command returns a `CommandError`.
pub fn make_test_services_with_command_error(
    command_name: &str,
    err: weft_commands::CommandError,
) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(
        provider,
        hooks,
        Some((command_name.to_string(), err)),
        None,
        false,
    )
}

/// Build test `MockServiceLocator` where a specific command returns `CommandResult { success: false }`.
pub fn make_test_services_with_failed_command(
    command_name: &str,
    error_msg: &str,
) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(
        provider,
        hooks,
        None,
        Some((command_name.to_string(), error_msg.to_string())),
        false,
    )
}

/// Build test `MockServiceLocator` with a hook runner that blocks the specified event.
pub fn make_test_services_with_blocking_hook(event: HookEvent, reason: &str) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(BlockingHookRunner {
        blocking_event: event,
        reason: reason.to_string(),
    });
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` whose provider yields delta chunks.
pub fn make_test_services_with_chunk_stream(
    chunks: Vec<String>,
    include_complete: bool,
) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(ChunkStreamProvider::new(chunks, include_complete));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` whose provider stream errors after the first chunk.
pub fn make_test_services_with_mid_stream_error(
    first_chunk: &str,
    err: ProviderError,
) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(MidStreamErrorProvider::new(first_chunk, err));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` whose semantic router always returns an error.
pub fn make_test_services_with_failing_router() -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);

    let config = Arc::new(make_test_config());
    let providers: Arc<dyn weft_llm_trait::ProviderService + Send + Sync> =
        Arc::new(StubProviderService::new(provider));
    let router: Arc<dyn weft_router_trait::SemanticRouter + Send + Sync> =
        Arc::new(ErrorRouter::new("test router failure"));
    let commands: Arc<dyn weft_commands_trait::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry::new());

    MockServiceLocator {
        config,
        providers,
        router,
        commands,
        hooks,
        memory: None,
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    }
}

/// Build test `MockServiceLocator` where `list_commands` returns an error.
pub fn make_test_services_with_failing_list_commands() -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, true)
}

/// Build test `MockServiceLocator` whose provider delays before responding.
pub fn make_test_services_with_slow_provider(
    delay_secs: u64,
    response_text: &str,
) -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(SlowProvider::new(delay_secs, response_text));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_mock(provider, hooks, None, None, false)
}

/// Build test `MockServiceLocator` with memory stores configured.
///
/// The config includes `[[memory.stores]]` so `services.config().memory` is Some.
/// The actual memory service field remains None — only the config matters for the
/// command_selection activity's `has_memory` check.
pub fn make_test_services_with_memory() -> MockServiceLocator {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);

    let config = Arc::new(make_test_config_with_memory());
    let providers: Arc<dyn weft_llm_trait::ProviderService + Send + Sync> =
        Arc::new(StubProviderService::new(provider));
    let router: Arc<dyn weft_router_trait::SemanticRouter + Send + Sync> = Arc::new(StubRouter);
    let commands: Arc<dyn weft_commands_trait::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry::new());

    MockServiceLocator {
        config,
        providers,
        router,
        commands,
        hooks,
        memory: None, // config presence is what matters, not the service
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    }
}

// ── ActivityInput builder ─────────────────────────────────────────────────────

/// Build a minimal valid `ActivityInput` for use in tests.
///
/// Contains one user message with text "hello", an auto routing instruction,
/// a stub routing result for "stub-model", a default budget, and null metadata.
pub fn make_test_input() -> ActivityInput {
    use chrono::Utc;

    ActivityInput {
        messages: vec![WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        }],
        request: WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        },
        routing_result: Some(RoutingSnapshot {
            model_routing: RoutingActivity {
                model: "stub-model".to_string(),
                score: 0.9,
                filters: vec![],
            },
            tool_necessity: None,
            tool_necessity_score: None,
        }),
        budget: Budget::new(10, 5, 3, Utc::now() + chrono::Duration::hours(1)),
        metadata: serde_json::Value::Null,
        generation_config: None,
        accumulated_text: String::new(),
        available_commands: vec![],
        idempotency_key: None,
        accumulated_usage: weft_core::WeftUsage::default(),
        child_spawner: None,
    }
}

// ── Event collection helper ───────────────────────────────────────────────────

/// Drain all events from an `mpsc::Receiver<PipelineEvent>` into a `Vec`.
///
/// Uses `try_recv` in a loop to collect all currently queued events without
/// blocking. Call this after the activity under test has completed.
pub fn collect_events(rx: &mut mpsc::Receiver<PipelineEvent>) -> Vec<PipelineEvent> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use weft_commands::CommandRegistry as _;

    #[test]
    fn make_test_services_returns_mock_service_locator() {
        let services = make_test_services();
        assert_eq!(services.providers().default_name(), "stub-model");
        assert!(services.memory.is_none());
    }

    #[test]
    fn make_test_input_has_user_message() {
        let input = make_test_input();
        assert!(!input.messages.is_empty());
        assert_eq!(input.messages[0].role, Role::User);
    }

    #[test]
    fn make_test_services_with_memory_has_memory_config() {
        let services = make_test_services_with_memory();
        let has_memory = services
            .config()
            .memory
            .as_ref()
            .is_some_and(|m| !m.stores.is_empty());
        assert!(has_memory, "memory config must have at least one store");
    }

    #[tokio::test]
    async fn blocking_hook_runner_blocks_configured_event() {
        let runner = BlockingHookRunner {
            blocking_event: HookEvent::RequestStart,
            reason: "test block".to_string(),
        };
        let result = runner
            .run_chain(HookEvent::RequestStart, serde_json::json!({}), None)
            .await;
        assert!(matches!(result, HookChainResult::Blocked { .. }));
    }

    #[tokio::test]
    async fn blocking_hook_runner_allows_other_events() {
        let runner = BlockingHookRunner {
            blocking_event: HookEvent::RequestStart,
            reason: "test block".to_string(),
        };
        let result = runner
            .run_chain(HookEvent::PreResponse, serde_json::json!({}), None)
            .await;
        assert!(matches!(result, HookChainResult::Allowed { .. }));
    }

    #[tokio::test]
    async fn collect_events_drains_channel() {
        use weft_reactor_trait::ExecutionEvent;
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
            .await
            .unwrap();
        tx.send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
            .await
            .unwrap();
        let events = collect_events(&mut rx);
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn stub_command_registry_lists_commands() {
        let registry = StubCommandRegistry::new();
        let cmds = registry.list_commands().await.unwrap();
        assert!(!cmds.is_empty());
    }
}
