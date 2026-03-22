//! Shared test infrastructure for weft_reactor integration tests.
//!
//! This module contains activity stubs and helper functions shared across
//! the integration test files. Items used only within a single test stay inline
//! in that test file.
//!
//! Dead code warnings are suppressed because each integration test file is a
//! separate compilation unit and only uses a subset of the items defined here.
#![allow(dead_code)]

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
use weft_reactor::event::{
    ActivityEvent, CommandEvent, CommandFormat, ContextEvent, ExecutionEvent, FailureDetail,
    GeneratedEvent, GenerationEvent, MessageInjectionSource, PipelineEvent, SelectionEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::registry::ActivityRegistry;
use weft_reactor::test_support::TestEventLog;

use weft_core::{
    ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftMessage, WeftRequest,
};

// ── Top-level activity stubs ──────────────────────────────────────────────

/// Activity that immediately pushes Done (no commands, no content).
pub struct ImmediateDoneActivity {
    pub name: String,
}

#[async_trait::async_trait]
impl Activity for ImmediateDoneActivity {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        if cancel.is_cancelled() {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name.clone(),
                    error: "cancelled".to_string(),
                    retryable: false,
                    detail: FailureDetail::default(),
                }))
                .await;
            return;
        }
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Started {
                model: "stub-model".to_string(),
                message_count: input.messages.len(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                GeneratedEvent::Done,
            )))
            .await;
        let response_message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(String::new())],
            delta: false,
            message_index: 0,
        };
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Completed {
                model: "stub-model".to_string(),
                response_message,
                generated_events: vec![GeneratedEvent::Done],
                input_tokens: Some(5),
                output_tokens: Some(0),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name.clone(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;
    }
}

/// Activity that pushes text content then Done.
pub struct TextGenerateActivity {
    pub name: String,
    pub response_text: String,
}

#[async_trait::async_trait]
impl Activity for TextGenerateActivity {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        if cancel.is_cancelled() {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name.clone(),
                    error: "cancelled".to_string(),
                    retryable: false,
                    detail: FailureDetail::default(),
                }))
                .await;
            return;
        }
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Started {
                model: "stub-model".to_string(),
                message_count: input.messages.len(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                GeneratedEvent::Content {
                    part: ContentPart::Text(self.response_text.clone()),
                },
            )))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                GeneratedEvent::Done,
            )))
            .await;
        let response_message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(self.response_text.clone())],
            delta: false,
            message_index: 0,
        };
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Completed {
                model: "stub-model".to_string(),
                response_message,
                generated_events: vec![GeneratedEvent::Done],
                input_tokens: Some(5),
                output_tokens: Some(3),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name.clone(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;
    }
}

/// Activity that always fails with a configurable retryable flag.
pub struct AlwaysFailActivity {
    pub name: String,
    pub retryable: bool,
    pub error_msg: String,
}

#[async_trait::async_trait]
impl Activity for AlwaysFailActivity {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Failed {
                name: self.name.clone(),
                error: self.error_msg.clone(),
                retryable: self.retryable,
                detail: FailureDetail::default(),
            }))
            .await;
    }
}

/// Activity that fails N times then succeeds on the (N+1)th call.
pub struct FailThenSucceedActivity {
    pub name: String,
    pub fails_remaining: std::sync::atomic::AtomicU32,
}

impl FailThenSucceedActivity {
    pub fn new(name: &str, fail_count: u32) -> Self {
        Self {
            name: name.to_string(),
            fails_remaining: std::sync::atomic::AtomicU32::new(fail_count),
        }
    }
}

#[async_trait::async_trait]
impl Activity for FailThenSucceedActivity {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;

        let remaining = self
            .fails_remaining
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |v| if v > 0 { Some(v - 1) } else { Some(0) },
            )
            .unwrap_or(0);

        if remaining > 0 {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name.clone(),
                    error: "transient failure".to_string(),
                    retryable: true,
                    detail: FailureDetail::default(),
                }))
                .await;
        } else {
            // Success.
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                    GeneratedEvent::Done,
                )))
                .await;
            let response_message = WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("stub-model".to_string()),
                content: vec![],
                delta: false,
                message_index: 0,
            };
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Completed {
                    model: "stub-model".to_string(),
                    response_message,
                    generated_events: vec![GeneratedEvent::Done],
                    input_tokens: None,
                    output_tokens: None,
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: self.name.clone(),
                    idempotency_key: input.idempotency_key.clone(),
                }))
                .await;
        }
    }
}

/// A no-op activity: immediately pushes ActivityCompleted (for validate, route, etc. stubs).
pub struct NoOpActivity {
    pub name: String,
}

#[async_trait::async_trait]
impl Activity for NoOpActivity {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name.clone(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// A minimal assemble_response activity stub.
pub struct StubAssembleResponse {
    pub name: String,
}

#[async_trait::async_trait]
impl Activity for StubAssembleResponse {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;

        let response_message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(input.accumulated_text.clone())],
            delta: false,
            message_index: 0,
        };
        let response = weft_core::WeftResponse {
            id: execution_id.to_string(),
            model: "stub-model".to_string(),
            messages: vec![response_message],
            usage: weft_core::WeftUsage::default(),
            timing: weft_core::WeftTiming::default(),
            degradations: vec![],
        };
        let _ = event_tx
            .send(PipelineEvent::Context(ContextEvent::ResponseAssembled {
                response,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name.clone(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// A no-op execute_command stub.
pub struct StubExecuteCommand {
    pub name: String,
}

#[async_trait::async_trait]
impl Activity for StubExecuteCommand {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        // Try to extract invocation from metadata.
        let cmd_name = input
            .metadata
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Command(CommandEvent::Completed {
                name: cmd_name.clone(),
                result: weft_core::CommandResult {
                    command_name: cmd_name,
                    success: true,
                    output: "stub output".to_string(),
                    error: None,
                },
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name.clone(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;
    }
}

// ── Pre-loop activity stubs ───────────────────────────────────────────────

/// Stub for ValidateActivity: emits ValidationPassed, CommandsAvailable, and Completed.
pub struct StubValidateActivity;

#[async_trait::async_trait]
impl Activity for StubValidateActivity {
    fn name(&self) -> &str {
        "validate"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Command(CommandEvent::Available {
                commands: vec![],
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// Stub for ModelSelectionActivity: emits ModelSelected with a non-empty model name.
pub struct StubModelSelectionActivity;

#[async_trait::async_trait]
impl Activity for StubModelSelectionActivity {
    fn name(&self) -> &str {
        "model_selection"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name: "stub-model".to_string(),
                score: 0.9,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// Stub for CommandSelectionActivity: emits CommandsSelected with an empty list.
pub struct StubCommandSelectionActivity;

#[async_trait::async_trait]
impl Activity for StubCommandSelectionActivity {
    fn name(&self) -> &str {
        "command_selection"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Selection(SelectionEvent::CommandsSelected {
                selected: vec![],
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// Stub for ProviderResolutionActivity: emits ProviderResolved with stub values.
pub struct StubProviderResolutionActivity;

#[async_trait::async_trait]
impl Activity for StubProviderResolutionActivity {
    fn name(&self) -> &str {
        "provider_resolution"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                model_name: "stub-model".to_string(),
                model_id: "stub-model-v1".to_string(),
                provider_name: "anthropic".to_string(),
                capabilities: vec![],
                max_tokens: 4096,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// Stub for SystemPromptAssemblyActivity: emits MessageInjected (SystemPromptAssembly)
/// and SystemPromptAssembled.
pub struct StubSystemPromptAssemblyActivity;

#[async_trait::async_trait]
impl Activity for StubSystemPromptAssemblyActivity {
    fn name(&self) -> &str {
        "system_prompt_assembly"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let system_message = weft_core::WeftMessage {
            role: weft_core::Role::System,
            source: weft_core::Source::Provider,
            model: None,
            content: vec![weft_core::ContentPart::Text("You are helpful.".to_string())],
            delta: false,
            message_index: 0,
        };
        let _ = event_tx
            .send(PipelineEvent::Context(ContextEvent::MessageInjected {
                message: system_message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Context(
                ContextEvent::SystemPromptAssembled { message_count: 1 },
            ))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// Stub for CommandFormattingActivity: emits CommandsFormatted with NoCommands.
pub struct StubCommandFormattingActivity;

#[async_trait::async_trait]
impl Activity for StubCommandFormattingActivity {
    fn name(&self) -> &str {
        "command_formatting"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Context(ContextEvent::CommandsFormatted {
                format: CommandFormat::NoCommands,
                command_count: 0,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

/// Stub for SamplingAdjustmentActivity: emits SamplingUpdated with max_tokens 4096.
pub struct StubSamplingAdjustmentActivity;

#[async_trait::async_trait]
impl Activity for StubSamplingAdjustmentActivity {
    fn name(&self) -> &str {
        "sampling_adjustment"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Context(ContextEvent::SamplingUpdated {
                max_tokens: 4096,
                temperature: None,
                top_p: None,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

// ── Degradation stubs (used by multiple test files) ───────────────────────

/// Activity that always fails (non-critical) — for testing degradation.
pub struct FailingNonCriticalActivity {
    pub activity_name: String,
    pub error_code: String,
}

#[async_trait::async_trait]
impl Activity for FailingNonCriticalActivity {
    fn name(&self) -> &str {
        &self.activity_name
    }

    fn criticality(&self) -> weft_reactor_trait::Criticality {
        weft_reactor_trait::Criticality::NonCritical
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.activity_name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Failed {
                name: self.activity_name.clone(),
                error: format!("{} failed", self.activity_name),
                retryable: false,
                detail: FailureDetail {
                    error_code: self.error_code.clone(),
                    detail: serde_json::Value::Null,
                    cause: None,
                    attempted: None,
                    fallback: None,
                },
            }))
            .await;
    }
}

/// Activity that always fails with SemiCritical criticality.
pub struct FailingSemiCriticalActivity {
    pub activity_name: String,
}

#[async_trait::async_trait]
impl Activity for FailingSemiCriticalActivity {
    fn name(&self) -> &str {
        &self.activity_name
    }

    fn criticality(&self) -> weft_reactor_trait::Criticality {
        weft_reactor_trait::Criticality::SemiCritical
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        _input: ActivityInput,
        _services: &dyn weft_reactor_trait::ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.activity_name.clone(),
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Failed {
                name: self.activity_name.clone(),
                error: format!("{} failed", self.activity_name),
                retryable: false,
                detail: FailureDetail {
                    error_code: "no_matching_model".to_string(),
                    detail: serde_json::Value::Null,
                    cause: None,
                    attempted: None,
                    fallback: None,
                },
            }))
            .await;
    }
}

// ── Reactor builder helpers ───────────────────────────────────────────────

/// Build a simple pipeline config with a single generate activity and no pre/post activities.
pub fn simple_pipeline_config(generate_name: &str) -> PipelineConfig {
    PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name(generate_name.to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    }
}

/// Build a pipeline config with validate in pre-loop.
pub fn pipeline_with_validate(generate_name: &str) -> PipelineConfig {
    PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![ActivityRef::Name("validate".to_string())],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name(generate_name.to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    }
}

/// Build a ReactorConfig with the given pipeline config.
pub fn reactor_config(pipeline: PipelineConfig) -> ReactorConfig {
    ReactorConfig {
        pipelines: vec![pipeline],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    }
}

/// Build a minimal WeftRequest for testing.
pub fn test_request() -> WeftRequest {
    WeftRequest {
        messages: vec![WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        }],
        routing: ModelRoutingInstruction::parse("auto"),
        options: SamplingOptions::default(),
    }
}

/// Build a registry with all the given activities registered.
pub fn build_registry(activities: Vec<Arc<dyn Activity>>) -> Arc<ActivityRegistry> {
    let mut registry = ActivityRegistry::new();
    for activity in activities {
        registry
            .register(activity)
            .expect("duplicate activity name in test");
    }
    Arc::new(registry)
}

/// Build a TestEventLog for testing.
pub fn test_event_log() -> Arc<TestEventLog> {
    Arc::new(TestEventLog::new())
}

/// Build a registry with the full pre-loop activity set.
pub fn build_new_preloop_registry(generate_name: &str) -> Arc<ActivityRegistry> {
    build_registry(vec![
        Arc::new(StubValidateActivity),
        Arc::new(StubModelSelectionActivity),
        Arc::new(StubCommandSelectionActivity),
        Arc::new(StubProviderResolutionActivity),
        Arc::new(StubSystemPromptAssemblyActivity),
        Arc::new(StubCommandFormattingActivity),
        Arc::new(StubSamplingAdjustmentActivity),
        Arc::new(ImmediateDoneActivity {
            name: generate_name.to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ])
}

/// Build a pipeline config with the full pre-loop activity set.
pub fn new_preloop_pipeline_config(generate_name: &str) -> PipelineConfig {
    PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![
            ActivityRef::Name("validate".to_string()),
            ActivityRef::Name("model_selection".to_string()),
            ActivityRef::Name("command_selection".to_string()),
            ActivityRef::Name("provider_resolution".to_string()),
            ActivityRef::Name("system_prompt_assembly".to_string()),
            ActivityRef::Name("command_formatting".to_string()),
            ActivityRef::Name("sampling_adjustment".to_string()),
        ],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name(generate_name.to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    }
}
