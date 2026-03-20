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

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use tokio::sync::mpsc;
use weft_core::{
    CommandDescription, CommandInvocation, CommandResult, CommandStub, ContentPart, HookEvent,
    ModelRoutingInstruction, Role, RoutingActivity, SamplingOptions, Source, WeftMessage,
    WeftRequest,
};
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::{
    Capability, ContentDelta, Provider, ProviderChunk, ProviderError, ProviderRequest,
    ProviderResponse, ProviderService, TokenUsage,
};
use weft_router::{
    RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate, SemanticRouter,
};

use crate::activity::ActivityInput;
use crate::budget::Budget;
use crate::event::PipelineEvent;
use crate::event_log::{EventLog, EventLogError};
use crate::execution::{Execution, ExecutionId, ExecutionStatus};
use crate::services::Services;

// ── Stub Provider ─────────────────────────────────────────────────────────────

/// A stub provider that always returns a fixed text response.
struct StubProvider {
    response_text: String,
}

#[async_trait::async_trait]
impl Provider for StubProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(self.response_text.clone())],
            delta: false,
            message_index: 0,
        };
        Ok(ProviderResponse::ChatCompletion {
            message,
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        })
    }

    fn name(&self) -> &str {
        "stub-provider"
    }
}

// ProviderError doesn't implement Clone, so we store the error as a string and
// re-construct a representative error for repeated calls. For test purposes,
// we store the original error in an Arc<Mutex<Option<ProviderError>>> and yield
// it once; subsequent calls return Unsupported. In practice, each test creates
// a fresh Services, so the provider is called at most once.
struct SingleUseErrorProvider {
    error: std::sync::Mutex<Option<ProviderError>>,
    fallback_msg: String,
}

#[async_trait::async_trait]
impl Provider for SingleUseErrorProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let err = self.error.lock().unwrap().take();
        match err {
            Some(e) => Err(e),
            None => Err(ProviderError::Unsupported(self.fallback_msg.clone())),
        }
    }

    fn name(&self) -> &str {
        "error-provider"
    }
}

/// A provider that sleeps for a configurable duration before returning a response.
///
/// Used in tests that need the provider call to take time (e.g., heartbeat tests).
/// Only useful with `tokio::time::pause()` + `tokio::time::advance()`.
pub struct SlowProvider {
    delay_secs: u64,
    response_text: String,
}

#[async_trait::async_trait]
impl Provider for SlowProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        tokio::time::sleep(tokio::time::Duration::from_secs(self.delay_secs)).await;
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(self.response_text.clone())],
            delta: false,
            message_index: 0,
        };
        Ok(ProviderResponse::ChatCompletion {
            message,
            usage: None,
        })
    }

    fn name(&self) -> &str {
        "slow-provider"
    }
}

/// A provider that yields a fixed sequence of text delta chunks, then a Complete.
///
/// Used in tests that verify `GenerateActivity` pushes `Generated(Content)` events
/// as each chunk arrives rather than buffering the full response.
pub struct ChunkStreamProvider {
    /// Text chunks to yield as `ProviderChunk::Delta` before the `Complete`.
    pub chunks: Vec<String>,
    /// Whether to include a `Complete` chunk at the end.
    ///
    /// When `true`, the last chunk is `Complete` carrying the concatenated text
    /// and usage data. When `false`, the stream ends after all delta chunks
    /// (simulates a provider that yields only deltas).
    pub include_complete: bool,
}

#[async_trait::async_trait]
impl Provider for ChunkStreamProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        // Assemble a response from all chunks for the non-streaming path.
        let text: String = self.chunks.join("");
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("chunk-stream-model".to_string()),
            content: vec![ContentPart::Text(text)],
            delta: false,
            message_index: 0,
        };
        Ok(ProviderResponse::ChatCompletion {
            message,
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: self.chunks.len() as u32 * 2,
            }),
        })
    }

    async fn execute_stream(
        &self,
        _request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        let chunks = self.chunks.clone();
        let include_complete = self.include_complete;

        // Collect all items into a Vec upfront, then convert to a stream.
        let mut items: Vec<Result<ProviderChunk, ProviderError>> = chunks
            .iter()
            .map(|text| {
                Ok(ProviderChunk::Delta(ContentDelta {
                    text: Some(text.clone()),
                    tool_call_delta: None,
                }))
            })
            .collect();

        if include_complete {
            let full_text: String = chunks.join("");
            let message = WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("chunk-stream-model".to_string()),
                content: vec![ContentPart::Text(full_text)],
                delta: false,
                message_index: 0,
            };
            let response = ProviderResponse::ChatCompletion {
                message,
                usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: chunks.len() as u32 * 2,
                }),
            };
            items.push(Ok(ProviderChunk::Complete(response)));
        }

        Ok(Box::pin(futures::stream::iter(items)))
    }

    fn name(&self) -> &str {
        "chunk-stream-provider"
    }
}

/// A provider whose stream yields one delta then errors.
///
/// Used in tests that verify `GenerateActivity` pushes `GenerationFailed` when
/// a mid-stream error occurs after partial content was already emitted.
pub struct MidStreamErrorProvider {
    /// The text chunk to emit before the error.
    pub first_chunk: String,
    /// The error to return as the second item.
    pub error: std::sync::Mutex<Option<ProviderError>>,
    pub fallback_error_msg: String,
}

#[async_trait::async_trait]
impl Provider for MidStreamErrorProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        Err(ProviderError::Unsupported(
            "mid-stream-error-provider: use execute_stream".to_string(),
        ))
    }

    async fn execute_stream(
        &self,
        _request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        let first = self.first_chunk.clone();
        let err = self
            .error
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| ProviderError::Unsupported(self.fallback_error_msg.clone()));

        let items: Vec<Result<ProviderChunk, ProviderError>> = vec![
            Ok(ProviderChunk::Delta(ContentDelta {
                text: Some(first),
                tool_call_delta: None,
            })),
            Err(err),
        ];
        Ok(Box::pin(futures::stream::iter(items)))
    }

    fn name(&self) -> &str {
        "mid-stream-error-provider"
    }
}

/// Build test `Services` whose provider delays for `delay_secs` before responding.
///
/// Used with `tokio::time::pause()` to test heartbeat emission.
pub fn make_test_services_with_slow_provider(delay_secs: u64, response_text: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(SlowProvider {
        delay_secs,
        response_text: response_text.to_string(),
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

// ── Stub ProviderService ──────────────────────────────────────────────────────

// ProviderService::models_with_capability returns &HashSet<String>, which requires
// the set to live long enough. We store an empty set in the service for the fallback.
struct StubProviderServiceV2 {
    provider: Arc<dyn Provider>,
    default: String,
    capabilities: HashMap<String, HashSet<Capability>>,
    capability_index: HashMap<Capability, HashSet<String>>,
    empty_string_set: HashSet<String>,
}

impl StubProviderServiceV2 {
    fn new(provider: Arc<dyn Provider>) -> Self {
        let default = "stub-model".to_string();
        let chat_cap = Capability::new(Capability::CHAT_COMPLETIONS);

        let mut capabilities = HashMap::new();
        let mut cap_set = HashSet::new();
        cap_set.insert(chat_cap.clone());
        capabilities.insert(default.clone(), cap_set);

        let mut capability_index: HashMap<Capability, HashSet<String>> = HashMap::new();
        let mut model_set = HashSet::new();
        model_set.insert(default.clone());
        capability_index.insert(chat_cap, model_set);

        Self {
            provider,
            default,
            capabilities,
            capability_index,
            empty_string_set: HashSet::new(),
        }
    }
}

impl ProviderService for StubProviderServiceV2 {
    fn get(&self, _name: &str) -> &Arc<dyn Provider> {
        // Return our single stub provider regardless of name.
        &self.provider
    }

    fn model_id(&self, name: &str) -> Option<&str> {
        if name == self.default {
            Some("stub-model-v1")
        } else {
            None
        }
    }

    fn max_tokens_for(&self, name: &str) -> Option<u32> {
        if name == self.default {
            Some(4096)
        } else {
            None
        }
    }

    fn default_provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    fn default_name(&self) -> &str {
        &self.default
    }

    fn models_with_capability(&self, capability: &Capability) -> &HashSet<String> {
        self.capability_index
            .get(capability)
            .unwrap_or(&self.empty_string_set)
    }

    fn model_has_capability(&self, model_name: &str, capability: &Capability) -> bool {
        self.capabilities
            .get(model_name)
            .map(|caps| caps.contains(capability))
            .unwrap_or(false)
    }

    fn model_capabilities(&self, model_name: &str) -> Option<&HashSet<Capability>> {
        self.capabilities.get(model_name)
    }
}

// ── Stub SemanticRouter ───────────────────────────────────────────────────────

/// A stub router that always selects "stub-model" with score 0.9.
struct StubRouter;

#[async_trait::async_trait]
impl SemanticRouter for StubRouter {
    async fn route(
        &self,
        _user_message: &str,
        domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, weft_router::RouterError> {
        let mut decision = RoutingDecision::empty();

        for (kind, candidates) in domains {
            match kind {
                RoutingDomainKind::Model => {
                    // Select the first candidate, or "stub-model" if none provided.
                    if let Some(first) = candidates.first() {
                        decision.model = Some(ScoredCandidate {
                            id: first.id.clone(),
                            score: 0.9,
                        });
                    } else {
                        decision.model = Some(ScoredCandidate {
                            id: "stub-model".to_string(),
                            score: 0.9,
                        });
                    }
                }
                RoutingDomainKind::Commands => {
                    // Score all candidates at 0.9.
                    decision.commands = candidates
                        .iter()
                        .map(|c| ScoredCandidate {
                            id: c.id.clone(),
                            score: 0.9,
                        })
                        .collect();
                }
                RoutingDomainKind::ToolNecessity => {
                    decision.tools_needed = Some(true);
                }
                RoutingDomainKind::Memory => {}
            }
        }

        Ok(decision)
    }

    async fn score_memory_candidates(
        &self,
        _text: &str,
        candidates: &[RoutingCandidate],
    ) -> Result<Vec<ScoredCandidate>, weft_router::RouterError> {
        Ok(candidates
            .iter()
            .map(|c| ScoredCandidate {
                id: c.id.clone(),
                score: 0.9,
            })
            .collect())
    }
}

// ── Error Router ─────────────────────────────────────────────────────────────

/// A stub router that always returns an error.
struct ErrorRouter {
    reason: String,
}

#[async_trait::async_trait]
impl SemanticRouter for ErrorRouter {
    async fn route(
        &self,
        _user_message: &str,
        _domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, weft_router::RouterError> {
        Err(weft_router::RouterError::InferenceFailed(
            self.reason.clone(),
        ))
    }

    async fn score_memory_candidates(
        &self,
        _text: &str,
        _candidates: &[RoutingCandidate],
    ) -> Result<Vec<ScoredCandidate>, weft_router::RouterError> {
        Err(weft_router::RouterError::InferenceFailed(
            self.reason.clone(),
        ))
    }
}

// ── Stub CommandRegistry ──────────────────────────────────────────────────────

/// A stub command registry. By default, all commands succeed.
/// Can be configured to fail with a specific error for a given command name,
/// or to return a `CommandResult { success: false }` for a given command name.
struct StubCommandRegistry {
    /// If set, `execute_command` for the matching name returns this error.
    failing_command: Option<(String, weft_commands::CommandError)>,
    /// If set, `execute_command` for the matching name returns a failed result (success=false).
    failed_result_command: Option<(String, String)>,
    /// If true, `list_commands` returns a `RegistryUnavailable` error.
    fail_list_commands: bool,
}

#[async_trait::async_trait]
impl weft_commands::CommandRegistry for StubCommandRegistry {
    async fn list_commands(&self) -> Result<Vec<CommandStub>, weft_commands::CommandError> {
        if self.fail_list_commands {
            return Err(weft_commands::CommandError::RegistryUnavailable(
                "stub: registry unavailable".to_string(),
            ));
        }
        Ok(vec![CommandStub {
            name: "test_command".to_string(),
            description: "A test command".to_string(),
        }])
    }

    async fn describe_command(
        &self,
        name: &str,
    ) -> Result<CommandDescription, weft_commands::CommandError> {
        Ok(CommandDescription {
            name: name.to_string(),
            description: format!("{name}: test command"),
            usage: format!("/{name}"),
            parameters_schema: None,
        })
    }

    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, weft_commands::CommandError> {
        // Check if this command is configured to fail with an infrastructure error.
        if let Some((ref failing_name, ref err)) = self.failing_command {
            if invocation.name == *failing_name {
                // We need to return the error. Since CommandError doesn't implement Clone,
                // we match on the stored error to construct a matching one.
                return Err(reconstruct_command_error(err));
            }
        }

        // Check if this command is configured to return a failed result (success=false).
        if let Some((ref failing_name, ref error_msg)) = self.failed_result_command {
            if invocation.name == *failing_name {
                return Ok(CommandResult {
                    command_name: invocation.name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(error_msg.clone()),
                });
            }
        }

        // Default: return a successful result.
        Ok(CommandResult {
            command_name: invocation.name.clone(),
            success: true,
            output: format!("stub output for {}", invocation.name),
            error: None,
        })
    }
}

/// Reconstruct a `CommandError` from a stored reference.
///
/// `CommandError` does not implement Clone. This helper reconstructs an equivalent
/// error from the stored value for use in test stubs where the error must be
/// returned from a `&self` method.
fn reconstruct_command_error(err: &weft_commands::CommandError) -> weft_commands::CommandError {
    match err {
        weft_commands::CommandError::NotFound(s) => {
            weft_commands::CommandError::NotFound(s.clone())
        }
        weft_commands::CommandError::ExecutionFailed { name, reason } => {
            weft_commands::CommandError::ExecutionFailed {
                name: name.clone(),
                reason: reason.clone(),
            }
        }
        weft_commands::CommandError::InvalidArguments { name, reason } => {
            weft_commands::CommandError::InvalidArguments {
                name: name.clone(),
                reason: reason.clone(),
            }
        }
        weft_commands::CommandError::RegistryUnavailable(s) => {
            weft_commands::CommandError::RegistryUnavailable(s.clone())
        }
    }
}

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
    let config = Arc::new(make_test_config());
    let providers: Arc<dyn weft_llm::ProviderService + Send + Sync> =
        Arc::new(StubProviderServiceV2::new(provider));
    let router: Arc<dyn weft_router::SemanticRouter + Send + Sync> = Arc::new(StubRouter);
    let commands: Arc<dyn weft_commands::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry {
            failing_command,
            failed_result_command,
            fail_list_commands,
        });

    Services {
        config,
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
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: "stub response".to_string(),
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider returns a fixed text response.
///
/// Use this when testing activities that interact with the provider
/// (e.g., `GenerateActivity`) and need a specific response text.
pub fn make_test_services_with_response(response_text: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: response_text.to_string(),
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider returns a `ProviderError`.
///
/// Use this when testing error paths in `GenerateActivity`.
pub fn make_test_services_with_error(err: ProviderError) -> Services {
    let error_msg = err.to_string();
    let provider: Arc<dyn Provider> = Arc::new(SingleUseErrorProvider {
        error: std::sync::Mutex::new(Some(err)),
        fallback_msg: error_msg,
    });
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
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: "stub response".to_string(),
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, Some((command_name.to_string(), err)))
}

/// Build test `Services` where executing a specific command returns a failed `CommandResult`.
///
/// The named command returns `CommandResult { success: false, error: Some(error_msg) }`.
/// All other commands return a successful result. Use this when testing the
/// `success: false` path in `ExecuteCommandActivity`.
pub fn make_test_services_with_failed_command(command_name: &str, error_msg: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: "stub response".to_string(),
    });
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
/// All other events are allowed. Use this when testing `HookActivity` block behavior
/// or `RouteActivity` pre/post hook blocking.
pub fn make_test_services_with_blocking_hook(event: HookEvent, reason: &str) -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: "stub response".to_string(),
    });
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
    let provider: Arc<dyn Provider> = Arc::new(ChunkStreamProvider {
        chunks,
        include_complete,
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose provider stream errors after the first chunk.
///
/// The first chunk pushes one `Generated(Content)` event, then the stream yields
/// an error. Use this to test `GenerateActivity` mid-stream error handling.
pub fn make_test_services_with_mid_stream_error(first_chunk: &str, err: ProviderError) -> Services {
    let error_msg = err.to_string();
    let provider: Arc<dyn Provider> = Arc::new(MidStreamErrorProvider {
        first_chunk: first_chunk.to_string(),
        error: std::sync::Mutex::new(Some(err)),
        fallback_error_msg: error_msg,
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services(provider, hooks, None)
}

/// Build test `Services` whose semantic router always returns an error.
///
/// Use this when testing error-fallback paths in selection activities:
/// - `ModelSelectionActivity` falls back to the default model on router error.
/// - `CommandSelectionActivity` falls back to all commands capped by `max_commands`.
pub fn make_test_services_with_failing_router() -> Services {
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: "stub response".to_string(),
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);

    let config = Arc::new(make_test_config());
    let providers: Arc<dyn weft_llm::ProviderService + Send + Sync> =
        Arc::new(StubProviderServiceV2::new(provider));
    let router: Arc<dyn weft_router::SemanticRouter + Send + Sync> = Arc::new(ErrorRouter {
        reason: "test router failure".to_string(),
    });
    let commands: Arc<dyn weft_commands::CommandRegistry + Send + Sync> =
        Arc::new(StubCommandRegistry {
            failing_command: None,
            failed_result_command: None,
            fail_list_commands: false,
        });

    Services {
        config,
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
    let provider: Arc<dyn Provider> = Arc::new(StubProvider {
        response_text: "stub response".to_string(),
    });
    let hooks: Arc<dyn HookRunner + Send + Sync> = Arc::new(weft_hooks::NullHookRunner);
    build_services_full_ext(provider, hooks, None, None, true)
}

// ── ActivityInput builder ─────────────────────────────────────────────────────

/// Build a minimal valid `ActivityInput` for use in tests.
///
/// Contains one user message with text "hello", an empty request, no routing
/// result, a default budget, null metadata, and no idempotency key.
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
        let services = make_test_services();
        assert_eq!(services.providers.default_name(), "stub-model");
        assert!(services.memory.is_none());
    }

    #[test]
    fn make_test_input_has_messages() {
        let input = make_test_input();
        assert!(!input.messages.is_empty());
        assert_eq!(input.messages[0].role, Role::User);
    }

    #[tokio::test]
    async fn stub_provider_returns_response() {
        let provider = StubProvider {
            response_text: "test response".to_string(),
        };
        let request = ProviderRequest::ChatCompletion {
            messages: vec![],
            model: "stub-model".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn error_provider_returns_error() {
        let provider = SingleUseErrorProvider {
            error: std::sync::Mutex::new(Some(ProviderError::RateLimited {
                retry_after_ms: 1000,
            })),
            fallback_msg: "rate limited".to_string(),
        };
        let request = ProviderRequest::ChatCompletion {
            messages: vec![],
            model: "stub-model".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn stub_router_selects_first_model_candidate() {
        let router = StubRouter;
        let candidates = vec![
            RoutingCandidate {
                id: "fast".to_string(),
                examples: vec!["quick answer".to_string()],
            },
            RoutingCandidate {
                id: "smart".to_string(),
                examples: vec!["complex reasoning".to_string()],
            },
        ];
        let decision = router
            .route("hello", &[(RoutingDomainKind::Model, candidates)])
            .await
            .unwrap();
        assert_eq!(decision.model.unwrap().id, "fast");
    }

    #[tokio::test]
    async fn stub_command_registry_lists_commands() {
        let registry = StubCommandRegistry {
            failing_command: None,
            failed_result_command: None,
            fail_list_commands: false,
        };
        let cmds = registry.list_commands().await.unwrap();
        assert!(!cmds.is_empty());
    }

    #[tokio::test]
    async fn stub_command_registry_executes_successfully() {
        let registry = StubCommandRegistry {
            failing_command: None,
            failed_result_command: None,
            fail_list_commands: false,
        };
        let invocation = CommandInvocation {
            name: "test_command".to_string(),
            action: weft_core::CommandAction::Execute,
            arguments: serde_json::json!({}),
        };
        let result = registry.execute_command(&invocation).await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn stub_command_registry_returns_error_for_configured_command() {
        let registry = StubCommandRegistry {
            failing_command: Some((
                "failing_cmd".to_string(),
                weft_commands::CommandError::RegistryUnavailable("down".to_string()),
            )),
            failed_result_command: None,
            fail_list_commands: false,
        };
        let invocation = CommandInvocation {
            name: "failing_cmd".to_string(),
            action: weft_core::CommandAction::Execute,
            arguments: serde_json::json!({}),
        };
        let result = registry.execute_command(&invocation).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            weft_commands::CommandError::RegistryUnavailable(_)
        ));
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
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(PipelineEvent::ValidationPassed).await.unwrap();
        tx.send(PipelineEvent::ValidationPassed).await.unwrap();
        let events = collect_events(&mut rx);
        assert_eq!(events.len(), 2);
    }
}
