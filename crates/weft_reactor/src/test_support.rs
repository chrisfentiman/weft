//! Test support utilities for weft_reactor.
//!
//! Provides builder functions for creating test `Services`, `ActivityInput`, and
//! event collection helpers. All items are gated behind `#[cfg(test)]`.
//!
//! **Available helpers:**
//! - [`make_test_services`] — Services with null stubs (NullHookRunner, stub provider/router/commands)
//! - [`make_test_services_with_response`] — Services whose provider returns a fixed text response
//! - [`make_test_services_with_error`] — Services whose provider returns a `ProviderError`
//! - [`make_test_services_with_command_error`] — Services where a specific command returns a `CommandError`
//! - [`make_test_services_with_blocking_hook`] — Services with a hook that blocks a specific event
//! - [`make_test_services_with_failing_list_commands`] — Services where `list_commands` returns an error
//! - [`make_test_input`] — Minimal valid `ActivityInput` for tests
//! - [`collect_events`] — Drain an `mpsc::Receiver<PipelineEvent>` into a `Vec`
//! - [`NullEventLog`] — No-op `EventLog` impl for use in activity tests
//!
//! **Why NullEventLog instead of InMemoryEventLog:**
//! `weft_eventlog_memory` has `weft_reactor` as a dependency. When cargo tests
//! the `weft_reactor` crate itself, two versions of `weft_reactor` appear in the
//! dependency graph (the crate under test vs the one `weft_eventlog_memory` was
//! compiled against), causing trait impl mismatches. The `NullEventLog` avoids
//! that circular dependency for activity tests.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use weft_core::{
    ContentPart, HookEvent, ModelRoutingInstruction, Role, RoutingActivity, SamplingOptions,
    Source, WeftMessage, WeftRequest,
};
use weft_hooks::{HookChainResult, HookRunner};
use weft_providers::{Provider, ProviderError};

// Import stub types from trait crates instead of defining locally.
pub use weft_commands::test_support::{StubCommandRegistry, reconstruct_command_error};
pub use weft_providers::test_support::{
    ChunkStreamProvider, MidStreamErrorProvider, SingleUseErrorProvider, SlowProvider,
    StubProvider, StubProviderService,
};
pub use weft_router::test_support::{ErrorRouter, StubRouter};

use crate::activity::ActivityInput;
use crate::budget::Budget;
use crate::event::PipelineEvent;
use crate::event_log::{EventLog, EventLogError};
use crate::execution::{Execution, ExecutionId, ExecutionStatus};
use crate::services::Services;

// ── Stub HookRunner ───────────────────────────────────────────────────────────

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

// ── NullEventLog ─────────────────────────────────────────────────────────────

/// A no-op [`EventLog`] implementation for activity tests.
///
/// All operations succeed and return empty/default values. Does not persist
/// any state. Use this instead of `weft_eventlog_memory::InMemoryEventLog`
/// to avoid a circular crate dependency in `weft_reactor` tests.
pub struct NullEventLog;

// ── TestEventLog ──────────────────────────────────────────────────────────────

/// An in-memory [`EventLog`] implementation for Reactor integration tests.
///
/// Persists events per execution in a `RwLock<HashMap>`. Supports idempotency
/// deduplication. Use this instead of `weft_eventlog_memory::InMemoryEventLog`
/// to avoid the circular crate dependency that arises when testing `weft_reactor`
/// itself.
pub struct TestEventLog {
    events: std::sync::RwLock<HashMap<ExecutionId, Vec<crate::event::Event>>>,
    executions: std::sync::RwLock<HashMap<ExecutionId, Execution>>,
}

impl TestEventLog {
    /// Create a new empty TestEventLog.
    pub fn new() -> Self {
        Self {
            events: std::sync::RwLock::new(HashMap::new()),
            executions: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Return all execution records currently stored, in arbitrary order.
    ///
    /// Used in tests to locate child executions and verify parent_id linkage.
    pub fn all_executions(&self) -> Vec<Execution> {
        let exec_map = self.executions.read().expect("poisoned lock");
        exec_map.values().cloned().collect()
    }
}

impl Default for TestEventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl EventLog for TestEventLog {
    async fn create_execution(&self, execution: &Execution) -> Result<(), EventLogError> {
        let mut exec_map = self.executions.write().expect("poisoned lock");
        let mut event_map = self.events.write().expect("poisoned lock");
        if exec_map.contains_key(&execution.id) {
            return Err(EventLogError::Storage(format!(
                "execution {} already exists",
                execution.id
            )));
        }
        exec_map.insert(execution.id.clone(), execution.clone());
        event_map.insert(execution.id.clone(), Vec::new());
        Ok(())
    }

    async fn update_execution_status(
        &self,
        execution_id: &ExecutionId,
        status: ExecutionStatus,
    ) -> Result<(), EventLogError> {
        let mut exec_map = self.executions.write().expect("poisoned lock");
        match exec_map.get_mut(execution_id) {
            Some(exec) => {
                exec.status = status;
                Ok(())
            }
            None => Err(EventLogError::ExecutionNotFound(execution_id.clone())),
        }
    }

    async fn append(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
        payload: serde_json::Value,
        schema_version: u32,
        idempotency_key: Option<&str>,
    ) -> Result<u64, EventLogError> {
        let mut event_map = self.events.write().expect("poisoned lock");
        let events = event_map
            .get_mut(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;

        // Idempotency: return existing sequence if key already present.
        if let Some(key) = idempotency_key {
            for existing in events.iter() {
                if existing
                    .payload
                    .get("idempotency_key")
                    .and_then(|v| v.as_str())
                    == Some(key)
                {
                    return Ok(existing.sequence);
                }
            }
        }

        let sequence = events.len() as u64 + 1;
        events.push(crate::event::Event {
            execution_id: execution_id.clone(),
            sequence,
            event_type: event_type.to_string(),
            payload,
            timestamp: chrono::Utc::now(),
            schema_version,
        });
        Ok(sequence)
    }

    async fn read(
        &self,
        execution_id: &ExecutionId,
        after_sequence: Option<u64>,
    ) -> Result<Vec<crate::event::Event>, EventLogError> {
        let event_map = self.events.read().expect("poisoned lock");
        let events = event_map
            .get(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;
        let result = match after_sequence {
            None => events.clone(),
            Some(n) => events.iter().filter(|e| e.sequence > n).cloned().collect(),
        };
        Ok(result)
    }

    async fn latest_of_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<Option<crate::event::Event>, EventLogError> {
        let event_map = self.events.read().expect("poisoned lock");
        let events = event_map
            .get(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;
        let result = events
            .iter()
            .rev()
            .find(|e| e.event_type == event_type)
            .cloned();
        Ok(result)
    }

    async fn count_by_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<u64, EventLogError> {
        let event_map = self.events.read().expect("poisoned lock");
        let events = event_map
            .get(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;
        Ok(events.iter().filter(|e| e.event_type == event_type).count() as u64)
    }
}

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
    ) -> Result<Vec<crate::event::Event>, EventLogError> {
        Ok(vec![])
    }

    async fn latest_of_type(
        &self,
        _execution_id: &ExecutionId,
        _event_type: &str,
    ) -> Result<Option<crate::event::Event>, EventLogError> {
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

// ── Services builders ─────────────────────────────────────────────────────────

/// Build minimal test `WeftConfig` for constructing Services.
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

/// Build `Services` using a given provider and hook runner.
fn build_services(
    provider: Arc<dyn Provider>,
    hooks: Arc<dyn HookRunner + Send + Sync>,
    failing_command: Option<(String, weft_commands::CommandError)>,
) -> Services {
    build_services_full(provider, hooks, failing_command, None)
}

/// Build `Services` with full control over stub command registry configuration.
fn build_services_full(
    provider: Arc<dyn Provider>,
    hooks: Arc<dyn HookRunner + Send + Sync>,
    failing_command: Option<(String, weft_commands::CommandError)>,
    failed_result_command: Option<(String, String)>,
) -> Services {
    build_services_full_ext(
        provider,
        hooks,
        failing_command,
        failed_result_command,
        false,
    )
}

/// Build `Services` with complete control including `list_commands` failure injection.
fn build_services_full_ext(
    provider: Arc<dyn Provider>,
    hooks: Arc<dyn HookRunner + Send + Sync>,
    failing_command: Option<(String, weft_commands::CommandError)>,
    failed_result_command: Option<(String, String)>,
    fail_list_commands: bool,
) -> Services {
    let weft_config = make_test_config();
    // StubProviderService (renamed from StubProviderServiceV2) comes from weft_providers::test_support.
    let providers: Arc<dyn weft_providers::ProviderService + Send + Sync> =
        Arc::new(StubProviderService::new(provider));
    // StubRouter comes from weft_router::test_support.
    let router: Arc<dyn weft_router::SemanticRouter + Send + Sync> = Arc::new(StubRouter);
    // StubCommandRegistry comes from weft_commands::test_support.
    let commands: Arc<dyn weft_commands::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry {
            failing_command,
            failed_result_command,
            fail_list_commands,
        });

    let config_store = Arc::new(weft_core::ConfigStore::new(weft_config));
    let resolved_config = config_store.snapshot();

    Services {
        config_store,
        resolved_config,
        providers,
        router,
        commands,
        memory: None,
        hooks,
        reactor_handle: std::sync::OnceLock::new(),
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    }
}

/// Build test `Services` with all null/stub implementations.
///
/// The provider returns "stub response" text. The hook runner allows all events.
/// The command registry returns a single stub command.
pub fn make_test_services() -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider returns a fixed text response.
///
/// Use this when testing activities that interact with the provider
/// (e.g., `GenerateActivity`) and need a specific response text.
pub fn make_test_services_with_response(response_text: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new(response_text));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider returns a `ProviderError`.
///
/// Use this when testing error paths in `GenerateActivity`.
pub fn make_test_services_with_error(err: ProviderError) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(SingleUseErrorProvider::new(err));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` where executing a specific command returns a `CommandError`.
///
/// The named command will fail with the provided error; all other commands succeed.
/// Use this when testing error paths in `ExecuteCommandActivity`.
pub fn make_test_services_with_command_error(
    command_name: &str,
    err: weft_commands::CommandError,
) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, Some((command_name.to_string(), err)))
}

/// Build test `Services` where executing a specific command returns a failed `CommandResult`.
///
/// The named command returns `CommandResult { success: false, error: Some(error_msg) }`.
/// All other commands return a successful result. Use this when testing the
/// `success: false` path in `ExecuteCommandActivity`.
pub fn make_test_services_with_failed_command(command_name: &str, error_msg: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services_full(
        provider,
        hooks,
        None,
        Some((command_name.to_string(), error_msg.to_string())),
    )
}

/// Build test `Services` with a hook runner that blocks the specified event.
///
/// All other events are allowed. Use this when testing `HookActivity` block behavior.
pub fn make_test_services_with_blocking_hook(event: HookEvent, reason: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(BlockingHookRunner {
        blocking_event: event,
        reason: reason.to_string(),
    });
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider yields `chunks` as streaming delta chunks.
///
/// When `include_complete` is `true`, the stream ends with a `Complete` chunk
/// containing the concatenated text and usage data. Use this to test that
/// `GenerateActivity` pushes one `Generated(Content)` event per chunk.
pub fn make_test_services_with_chunk_stream(
    chunks: Vec<String>,
    include_complete: bool,
) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(ChunkStreamProvider::new(chunks, include_complete));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider stream errors after the first chunk.
///
/// The first chunk pushes one `Generated(Content)` event, then the stream yields
/// an error. Use this to test `GenerateActivity` mid-stream error handling.
pub fn make_test_services_with_mid_stream_error(first_chunk: &str, err: ProviderError) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(MidStreamErrorProvider::new(first_chunk, err));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose semantic router always returns an error.
///
/// Use this when testing error-fallback paths in selection activities:
/// - `ModelSelectionActivity` falls back to the default model on router error.
/// - `CommandSelectionActivity` falls back to all commands capped by `max_commands`.
pub fn make_test_services_with_failing_router() -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);

    let weft_config = make_test_config();
    let providers: Arc<dyn weft_providers::ProviderService + Send + Sync> =
        Arc::new(StubProviderService::new(provider));
    // ErrorRouter comes from weft_router::test_support.
    let router: Arc<dyn weft_router::SemanticRouter + Send + Sync> =
        Arc::new(ErrorRouter::new("test router failure"));
    let commands: Arc<dyn weft_commands::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry::new());

    let config_store = Arc::new(weft_core::ConfigStore::new(weft_config));
    let resolved_config = config_store.snapshot();

    Services {
        config_store,
        resolved_config,
        providers,
        router,
        commands,
        memory: None,
        hooks,
        reactor_handle: std::sync::OnceLock::new(),
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    }
}

/// Build test `Services` whose semantic router always returns a `RegistryUnavailable` error.
pub fn make_test_services_with_failing_list_commands() -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("stub response"));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services_full_ext(provider, hooks, None, None, true)
}

/// Build test `Services` whose provider delays for `delay_secs` before responding.
///
/// Used with `tokio::time::pause()` to test heartbeat emission.
pub fn make_test_services_with_slow_provider(delay_secs: u64, response_text: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(SlowProvider::new(delay_secs, response_text));
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

// ── ActivityInput builder ─────────────────────────────────────────────────────

/// Build a minimal valid `ActivityInput` for use in tests.
///
/// Contains one user message with text "hello", an empty request, no routing
/// result, a default budget, null metadata, and no idempotency key.
pub fn make_test_input() -> ActivityInput {
    use chrono::Utc;

    let weft_config = make_test_config();
    let config_store = weft_core::ConfigStore::new(weft_config);
    let config = config_store.snapshot();

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
        routing_result: Some(crate::activity::RoutingSnapshot {
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
        config,
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

    // ── Smoke tests for test_support helpers ──────────────────────────────

    #[test]
    fn make_test_services_returns_services() {
        use weft_reactor_trait::ServiceLocator as _;
        let services = make_test_services();
        assert_eq!(services.providers().default_name(), "stub-model");
        assert!(services.memory.is_none());
    }

    #[test]
    fn make_test_input_has_messages() {
        let input = make_test_input();
        assert!(!input.messages.is_empty());
        assert_eq!(input.messages[0].role, Role::User);
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
        use crate::event::ExecutionEvent;
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
    async fn stub_command_registry_from_import_lists_commands() {
        let registry = StubCommandRegistry::new();
        let cmds = registry.list_commands().await.unwrap();
        assert!(!cmds.is_empty());
    }
}
