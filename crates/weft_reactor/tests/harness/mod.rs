//! Shared test infrastructure for weft_reactor integration tests.
//!
//! This module provides `TestActivity`, a composable builder that replaces
//! hand-written activity stubs. All shared helpers are here; inline stubs
//! in individual test files have been eliminated.
//!
//! Dead code warnings are suppressed because each integration test file is a
//! separate compilation unit and only uses a subset of the items defined here.
#![allow(dead_code)]

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, Ordering},
};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_core::{
    CommandAction, CommandInvocation, ContentPart, ModelRoutingInstruction, Role, SamplingOptions,
    Source, WeftMessage, WeftRequest,
};
use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
use weft_reactor::event::{
    ActivityEvent, CommandEvent, CommandFormat, ContextEvent, ExecutionEvent, FailureDetail,
    GeneratedEvent, GenerationEvent, HookOutcome, MessageInjectionSource, PipelineEvent,
    SelectionEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::registry::ActivityRegistry;
use weft_reactor::test_support::TestEventLog;
use weft_reactor_trait::{Criticality, ServiceLocator};

// ── TestActivity ──────────────────────────────────────────────────────────────

/// A configurable test activity that replaces hand-written stubs.
///
/// Behavior is determined by `Behavior` variants. The struct implements
/// `Activity` and dispatches to the configured behavior in `execute()`.
pub struct TestActivity {
    pub activity_name: String,
    pub criticality: Criticality,
    pub behavior: Behavior,
}

/// What the activity does when executed.
pub enum Behavior {
    /// Emit Started, Generation(Started, Content?, Done, Completed), Activity(Completed).
    Generate {
        response_text: Option<String>,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },

    /// Emit Started, then immediately Failed.
    Fail {
        error: String,
        retryable: bool,
        detail: FailureDetail,
    },

    /// Fail N times (tracked by shared counter), then succeed with Generate behavior.
    FailThenSucceed { fail_count: Arc<AtomicU32> },

    /// Emit Started, a list of domain events, Completed. For pre-loop stubs.
    EmitEvents { events: Vec<PipelineEvent> },

    /// Emit Started, assemble a WeftResponse from input accumulated_text, Completed.
    AssembleResponse,

    /// Emit Started, Command(Completed), Activity(Completed).
    ExecuteCommand,

    /// Emit Started, wait for cancellation, then emit Failed.
    WaitForCancel { pre_stall: PreStallBehavior },

    /// Emit Signal::Cancel on the event channel, optionally sleep afterward.
    EmitCancel {
        reason: String,
        post_sleep: Option<Duration>,
    },

    /// Call-count-aware behavior: different actions per invocation.
    PerCall {
        call_count: Arc<AtomicU32>,
        /// Actions indexed by call number (0-based). Calls beyond the vec use `default_action`.
        actions: Vec<CallAction>,
        default_action: CallAction,
    },

    /// Run hook chain and emit the result.
    RunHook {
        hook_event: weft_core::HookEvent,
        /// String name used in HookOutcome::Blocked (e.g. "request_start").
        hook_event_name: String,
    },

    /// Block once then allow. For hook retry testing.
    HookBlockOnce {
        hook_event: String,
        hook_name: String,
        block_reason: String,
        call_count: Arc<AtomicU32>,
    },

    /// Capture a metadata field into shared state, then emit events.
    CaptureAndEmit {
        capture_field: String,
        captured: Arc<Mutex<Option<String>>>,
        events: Vec<PipelineEvent>,
    },

    /// NoOp: Started, ValidationPassed, Completed.
    NoOp,
}

/// What to do on a specific call number in a multi-call activity.
pub enum CallAction {
    /// Emit Generation events and Done. No commands.
    Done,

    /// Emit Generation events with command invocations, then Done.
    WithCommands { commands: Vec<String> },

    /// Emit Generation events, check for error text in messages (second-call pattern).
    UnlessErrorSeen,
}

/// Behavior before stalling on cancellation.
pub enum PreStallBehavior {
    /// No events before stalling.
    None,
    /// Emit N heartbeat events at the given interval, then stall.
    Heartbeats { count: usize, interval: Duration },
    /// Emit one content chunk, then stall.
    OneChunk { text: String },
}

// ── Activity impl ─────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl Activity for TestActivity {
    fn name(&self) -> &str {
        &self.activity_name
    }

    fn criticality(&self) -> Criticality {
        self.criticality
    }

    async fn execute(
        &self,
        execution_id: &ExecutionId,
        input: ActivityInput,
        services: &dyn ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let name = self.activity_name.clone();

        match &self.behavior {
            Behavior::Generate {
                response_text,
                input_tokens,
                output_tokens,
            } => {
                if cancel.is_cancelled() {
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Failed {
                            name: name.clone(),
                            error: "cancelled".to_string(),
                            retryable: false,
                            detail: FailureDetail::default(),
                        }))
                        .await;
                    return;
                }
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Started {
                        model: "stub-model".to_string(),
                        message_count: input.messages.len(),
                    }))
                    .await;
                if let Some(text) = response_text {
                    let _ = event_tx
                        .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                            GeneratedEvent::Content {
                                part: ContentPart::Text(text.clone()),
                            },
                        )))
                        .await;
                }
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::Done,
                    )))
                    .await;
                let content = response_text
                    .as_deref()
                    .map(|t| vec![ContentPart::Text(t.to_string())])
                    .unwrap_or_default();
                let response_message = WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: Some("stub-model".to_string()),
                    content,
                    delta: false,
                    message_index: 0,
                };
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Completed {
                        model: "stub-model".to_string(),
                        response_message,
                        generated_events: vec![GeneratedEvent::Done],
                        input_tokens: *input_tokens,
                        output_tokens: *output_tokens,
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Completed {
                        name: name.clone(),
                        idempotency_key: input.idempotency_key.clone(),
                    }))
                    .await;
            }

            Behavior::Fail {
                error,
                retryable,
                detail,
            } => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: name.clone(),
                        error: error.clone(),
                        retryable: *retryable,
                        detail: detail.clone(),
                    }))
                    .await;
            }

            Behavior::FailThenSucceed { fail_count } => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;

                // Decrement: if old value > 0, still failing.
                let old = fail_count.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                    if v > 0 { Some(v - 1) } else { Some(0) }
                });
                let remaining = old.unwrap_or(0);

                if remaining > 0 {
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Failed {
                            name: name.clone(),
                            error: "transient failure".to_string(),
                            retryable: true,
                            detail: FailureDetail::default(),
                        }))
                        .await;
                } else {
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
                            name: name.clone(),
                            idempotency_key: input.idempotency_key.clone(),
                        }))
                        .await;
                }
            }

            Behavior::EmitEvents { events } => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                for event in events.clone() {
                    let _ = event_tx.send(event).await;
                }
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Completed {
                        name: name.clone(),
                        idempotency_key: None,
                    }))
                    .await;
            }

            Behavior::AssembleResponse => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
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
                        name: name.clone(),
                        idempotency_key: None,
                    }))
                    .await;
            }

            Behavior::ExecuteCommand => {
                let cmd_name = input
                    .metadata
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
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
                        name: name.clone(),
                        idempotency_key: input.idempotency_key.clone(),
                    }))
                    .await;
            }

            Behavior::WaitForCancel { pre_stall } => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                match pre_stall {
                    PreStallBehavior::None => {}
                    PreStallBehavior::Heartbeats { count, interval } => {
                        for _ in 0..*count {
                            tokio::time::sleep(*interval).await;
                            let _ = event_tx
                                .send(PipelineEvent::Activity(ActivityEvent::Heartbeat {
                                    activity_name: name.clone(),
                                }))
                                .await;
                        }
                    }
                    PreStallBehavior::OneChunk { text } => {
                        let _ = event_tx
                            .send(PipelineEvent::Generation(GenerationEvent::Started {
                                model: "stub-model".to_string(),
                                message_count: input.messages.len(),
                            }))
                            .await;
                        let _ = event_tx
                            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                                GeneratedEvent::Content {
                                    part: ContentPart::Text(text.clone()),
                                },
                            )))
                            .await;
                    }
                }
                cancel.cancelled().await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: name.clone(),
                        error: "cancelled".to_string(),
                        retryable: false,
                        detail: FailureDetail::default(),
                    }))
                    .await;
            }

            Behavior::EmitCancel { reason, post_sleep } => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Signal(
                        weft_reactor::event::SignalEvent::Received(
                            weft_reactor::signal::Signal::Cancel {
                                reason: reason.clone(),
                            },
                        ),
                    ))
                    .await;
                if let Some(dur) = post_sleep {
                    tokio::time::sleep(*dur).await;
                }
            }

            Behavior::PerCall {
                call_count,
                actions,
                default_action,
            } => {
                let call_n = call_count.fetch_add(1, Ordering::SeqCst) as usize;
                let action = actions.get(call_n).unwrap_or(default_action);

                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Started {
                        model: "stub-model".to_string(),
                        message_count: input.messages.len(),
                    }))
                    .await;

                match action {
                    CallAction::Done => {}
                    CallAction::WithCommands { commands } => {
                        for cmd in commands {
                            let _ = event_tx
                                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                                    GeneratedEvent::CommandInvocation(CommandInvocation {
                                        name: cmd.clone(),
                                        action: CommandAction::Execute,
                                        arguments: serde_json::json!({}),
                                    }),
                                )))
                                .await;
                        }
                    }
                    CallAction::UnlessErrorSeen => {
                        // Only invoke command if no error text is present in messages.
                        let has_error = input.messages.iter().any(|m| {
                            m.content
                                .iter()
                                .any(|c| matches!(c, ContentPart::Text(t) if t.contains("failed")))
                        });
                        if !has_error {
                            // Use a hardcoded command name — tests using this variant set it via context.
                            // We check the first action's commands if available; otherwise use a fallback.
                            // This is for InvokeOnceActivity pattern: invoke a single command once.
                            let cmd_name = actions
                                .iter()
                                .find_map(|a| {
                                    if let CallAction::WithCommands { commands } = a {
                                        commands.first().cloned()
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or_else(|| "slow_tool".to_string());
                            let _ = event_tx
                                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                                    GeneratedEvent::CommandInvocation(CommandInvocation {
                                        name: cmd_name,
                                        action: CommandAction::Execute,
                                        arguments: serde_json::json!({}),
                                    }),
                                )))
                                .await;
                        }
                    }
                }

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
                        name: name.clone(),
                        idempotency_key: input.idempotency_key.clone(),
                    }))
                    .await;
            }

            Behavior::RunHook {
                hook_event,
                hook_event_name,
            } => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;

                let result = services
                    .hooks()
                    .run_chain(*hook_event, serde_json::json!({}), None)
                    .await;

                match result {
                    weft_hooks::HookChainResult::Blocked { hook_name, reason } => {
                        let _ = event_tx
                            .send(PipelineEvent::Hook(HookOutcome::Blocked {
                                hook_event: hook_event_name.clone(),
                                hook_name,
                                reason,
                            }))
                            .await;
                    }
                    weft_hooks::HookChainResult::Allowed { .. } => {
                        let _ = event_tx
                            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                                name: name.clone(),
                                idempotency_key: None,
                            }))
                            .await;
                    }
                }
            }

            Behavior::HookBlockOnce {
                hook_event,
                hook_name,
                block_reason,
                call_count,
            } => {
                let call_n = call_count.fetch_add(1, Ordering::SeqCst);

                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;

                if call_n == 0 {
                    let _ = event_tx
                        .send(PipelineEvent::Hook(HookOutcome::Blocked {
                            hook_event: hook_event.clone(),
                            hook_name: hook_name.clone(),
                            reason: block_reason.clone(),
                        }))
                        .await;
                } else {
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Completed {
                            name: name.clone(),
                            idempotency_key: input.idempotency_key.clone(),
                        }))
                        .await;
                }
            }

            Behavior::CaptureAndEmit {
                capture_field,
                captured,
                events,
            } => {
                let selected = input
                    .metadata
                    .get(capture_field.as_str())
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                *captured.lock().unwrap() = selected;

                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                for event in events.clone() {
                    let _ = event_tx.send(event).await;
                }
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Completed {
                        name: name.clone(),
                        idempotency_key: None,
                    }))
                    .await;
            }

            Behavior::NoOp => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Started {
                        name: name.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Completed {
                        name: name.clone(),
                        idempotency_key: None,
                    }))
                    .await;
            }
        }
    }
}

impl From<TestActivity> for Arc<dyn Activity> {
    fn from(a: TestActivity) -> Self {
        Arc::new(a)
    }
}

// ── Builder entry points ──────────────────────────────────────────────────────

impl TestActivity {
    /// Start building a generate activity.
    pub fn generate(name: &str) -> GenerateBuilder {
        GenerateBuilder {
            name: name.to_string(),
            criticality: Criticality::Critical,
            response_text: None,
            input_tokens: Some(5u32),
            output_tokens: Some(0u32),
        }
    }

    /// Start building a pre-loop activity that emits specific events.
    pub fn emitting(name: &str) -> EmittingBuilder {
        EmittingBuilder {
            name: name.to_string(),
            criticality: Criticality::Critical,
            events: vec![],
        }
    }

    /// Start building an always-failing activity.
    pub fn failing(name: &str) -> FailBuilder {
        FailBuilder {
            name: name.to_string(),
            criticality: Criticality::Critical,
            error: format!("{name} failed"),
            retryable: false,
            detail: FailureDetail::default(),
        }
    }

    /// Quick: assemble_response stub.
    pub fn assemble_response() -> Self {
        TestActivity {
            activity_name: "assemble_response".to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::AssembleResponse,
        }
    }

    /// Quick: execute_command stub.
    pub fn execute_command() -> Self {
        TestActivity {
            activity_name: "execute_command".to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::ExecuteCommand,
        }
    }

    /// Quick: no-op activity (Started, ValidationPassed, Completed).
    pub fn noop(name: &str) -> Self {
        TestActivity {
            activity_name: name.to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::NoOp,
        }
    }

    /// Quick: wait-for-cancel stub (no pre-stall events).
    pub fn stalling(name: &str) -> Self {
        TestActivity {
            activity_name: name.to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::WaitForCancel {
                pre_stall: PreStallBehavior::None,
            },
        }
    }

    /// Quick: wait-for-cancel with heartbeats before stalling.
    pub fn stalling_with_heartbeats(name: &str, count: usize, interval: Duration) -> Self {
        TestActivity {
            activity_name: name.to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::WaitForCancel {
                pre_stall: PreStallBehavior::Heartbeats { count, interval },
            },
        }
    }

    /// Quick: wait-for-cancel after emitting one content chunk.
    pub fn stalling_after_chunk(name: &str, text: &str) -> Self {
        TestActivity {
            activity_name: name.to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::WaitForCancel {
                pre_stall: PreStallBehavior::OneChunk {
                    text: text.to_string(),
                },
            },
        }
    }

    /// Quick: cancel-emitting stub.
    pub fn cancelling(name: &str, reason: &str) -> Self {
        TestActivity {
            activity_name: name.to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::EmitCancel {
                reason: reason.to_string(),
                post_sleep: None,
            },
        }
    }

    /// Quick: cancel-emitting stub with a post-sleep.
    pub fn cancelling_with_sleep(name: &str, reason: &str, sleep: Duration) -> Self {
        TestActivity {
            activity_name: name.to_string(),
            criticality: Criticality::Critical,
            behavior: Behavior::EmitCancel {
                reason: reason.to_string(),
                post_sleep: Some(sleep),
            },
        }
    }

    // ── Pre-loop stub convenience methods ─────────────────────────────────────

    /// Stub for validate: emits ValidationPassed + CommandsAvailable.
    pub fn validate_stub() -> Arc<dyn Activity> {
        TestActivity::emitting("validate")
            .event(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
            .event(PipelineEvent::Command(CommandEvent::Available {
                commands: vec![],
            }))
            .build()
    }

    /// Stub for model_selection: emits ModelSelected with stub-model.
    pub fn model_selection_stub() -> Arc<dyn Activity> {
        TestActivity::emitting("model_selection")
            .event(PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name: "stub-model".to_string(),
                score: 0.9,
            }))
            .build()
    }

    /// Stub for command_selection: emits CommandsSelected (empty).
    pub fn command_selection_stub() -> Arc<dyn Activity> {
        TestActivity::emitting("command_selection")
            .event(PipelineEvent::Selection(SelectionEvent::CommandsSelected {
                selected: vec![],
            }))
            .build()
    }

    /// Stub for provider_resolution: emits ProviderResolved with stub values.
    pub fn provider_resolution_stub() -> Arc<dyn Activity> {
        TestActivity::emitting("provider_resolution")
            .event(PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                model_name: "stub-model".to_string(),
                model_id: "stub-model-v1".to_string(),
                provider_name: "anthropic".to_string(),
                capabilities: vec![],
                max_tokens: 4096,
            }))
            .build()
    }

    /// Stub for system_prompt_assembly: emits MessageInjected + SystemPromptAssembled.
    pub fn system_prompt_assembly_stub() -> Arc<dyn Activity> {
        let system_message = WeftMessage {
            role: Role::System,
            source: Source::Provider,
            model: None,
            content: vec![ContentPart::Text("You are helpful.".to_string())],
            delta: false,
            message_index: 0,
        };
        TestActivity::emitting("system_prompt_assembly")
            .event(PipelineEvent::Context(ContextEvent::MessageInjected {
                message: system_message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }))
            .event(PipelineEvent::Context(
                ContextEvent::SystemPromptAssembled { message_count: 1 },
            ))
            .build()
    }

    /// Stub for command_formatting: emits CommandsFormatted (NoCommands).
    pub fn command_formatting_stub() -> Arc<dyn Activity> {
        TestActivity::emitting("command_formatting")
            .event(PipelineEvent::Context(ContextEvent::CommandsFormatted {
                format: CommandFormat::NoCommands,
                command_count: 0,
            }))
            .build()
    }

    /// Stub for sampling_adjustment: emits SamplingUpdated (max_tokens 4096).
    pub fn sampling_adjustment_stub() -> Arc<dyn Activity> {
        TestActivity::emitting("sampling_adjustment")
            .event(PipelineEvent::Context(ContextEvent::SamplingUpdated {
                max_tokens: 4096,
                temperature: None,
                top_p: None,
            }))
            .build()
    }
}

// ── GenerateBuilder ───────────────────────────────────────────────────────────

pub struct GenerateBuilder {
    name: String,
    criticality: Criticality,
    response_text: Option<String>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

impl GenerateBuilder {
    /// Set response text (default: None — no content chunk emitted).
    pub fn with_text(mut self, text: &str) -> Self {
        self.response_text = Some(text.to_string());
        self
    }

    /// Set token counts.
    pub fn with_tokens(mut self, input: u32, output: u32) -> Self {
        self.input_tokens = Some(input);
        self.output_tokens = Some(output);
        self
    }

    /// Switch to fail-then-succeed behavior. Returns Arc directly.
    pub fn fails_then_succeeds(self, fail_count: u32) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::FailThenSucceed {
                fail_count: Arc::new(AtomicU32::new(fail_count)),
            },
        })
    }

    /// Switch to per-call behavior: first call invokes commands, then generates Done.
    pub fn invokes_commands(self, commands: Vec<&str>) -> PerCallBuilder {
        PerCallBuilder {
            name: self.name,
            criticality: self.criticality,
            call_count: Arc::new(AtomicU32::new(0)),
            first_call_commands: commands.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Set criticality (default: Critical).
    pub fn with_criticality(mut self, c: Criticality) -> Self {
        self.criticality = c;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::Generate {
                response_text: self.response_text,
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
            },
        })
    }
}

// ── PerCallBuilder ────────────────────────────────────────────────────────────

pub struct PerCallBuilder {
    name: String,
    criticality: Criticality,
    call_count: Arc<AtomicU32>,
    first_call_commands: Vec<String>,
}

impl PerCallBuilder {
    /// Inject a shared call counter so tests can observe call counts.
    pub fn with_call_count(mut self, counter: Arc<AtomicU32>) -> Self {
        self.call_count = counter;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::PerCall {
                call_count: self.call_count,
                actions: vec![CallAction::WithCommands {
                    commands: self.first_call_commands,
                }],
                default_action: CallAction::Done,
            },
        })
    }
}

// ── EmittingBuilder ───────────────────────────────────────────────────────────

pub struct EmittingBuilder {
    name: String,
    criticality: Criticality,
    events: Vec<PipelineEvent>,
}

impl EmittingBuilder {
    /// Add a PipelineEvent to emit between Started and Completed.
    pub fn event(mut self, event: PipelineEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Set criticality.
    pub fn with_criticality(mut self, c: Criticality) -> Self {
        self.criticality = c;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::EmitEvents {
                events: self.events,
            },
        })
    }
}

// ── FailBuilder ───────────────────────────────────────────────────────────────

pub struct FailBuilder {
    name: String,
    criticality: Criticality,
    error: String,
    retryable: bool,
    detail: FailureDetail,
}

impl FailBuilder {
    /// Set error message.
    pub fn with_error(mut self, msg: &str) -> Self {
        self.error = msg.to_string();
        self
    }

    /// Set retryable flag.
    pub fn retryable(mut self, r: bool) -> Self {
        self.retryable = r;
        self
    }

    /// Set criticality.
    pub fn with_criticality(mut self, c: Criticality) -> Self {
        self.criticality = c;
        self
    }

    /// Set FailureDetail fields.
    pub fn with_detail(mut self, detail: FailureDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::Fail {
                error: self.error,
                retryable: self.retryable,
                detail: self.detail,
            },
        })
    }
}

// ── Failing execute_command builders ─────────────────────────────────────────

/// Build a failing execute_command activity (NonCritical, emits Failed).
pub fn failing_execute_command(error: &str, detail: FailureDetail) -> Arc<dyn Activity> {
    Arc::new(TestActivity {
        activity_name: "execute_command".to_string(),
        criticality: Criticality::NonCritical,
        behavior: Behavior::Fail {
            error: error.to_string(),
            retryable: false,
            detail,
        },
    })
}

/// Build a hanging execute_command activity (NonCritical, waits for cancel).
pub fn hanging_execute_command() -> Arc<dyn Activity> {
    Arc::new(TestActivity {
        activity_name: "execute_command".to_string(),
        criticality: Criticality::NonCritical,
        behavior: Behavior::WaitForCancel {
            pre_stall: PreStallBehavior::None,
        },
    })
}

// ── Registry convenience helpers ──────────────────────────────────────────────

/// Registry with generate + assemble_response + execute_command.
pub fn simple_registry(generate: Arc<dyn Activity>) -> Arc<ActivityRegistry> {
    build_registry(vec![
        generate,
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ])
}

/// Registry with full pre-loop pipeline + generate + assemble_response + execute_command.
pub fn preloop_registry(generate: Arc<dyn Activity>) -> Arc<ActivityRegistry> {
    build_registry(vec![
        TestActivity::validate_stub(),
        TestActivity::model_selection_stub(),
        TestActivity::command_selection_stub(),
        TestActivity::provider_resolution_stub(),
        TestActivity::system_prompt_assembly_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        generate,
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ])
}

// ── Reactor builder helpers ───────────────────────────────────────────────────

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
    preloop_registry(TestActivity::generate(generate_name).build())
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
