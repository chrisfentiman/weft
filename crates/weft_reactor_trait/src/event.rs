//! Pipeline event types and the event storage format.
//!
//! # Grouped structure
//!
//! `PipelineEvent` is a 10-variant outer enum. Each outer variant wraps a
//! category-specific inner enum with typed sub-variants. The total is 39 leaf
//! variants across the two levels.
//!
//! # Serde format
//!
//! Outer: `#[serde(tag = "category", content = "event")]` (adjacently tagged).
//! Inner: `#[serde(tag = "type")]` (internally tagged).
//!
//! ```json
//! {
//!   "category": "Activity",
//!   "event": {
//!     "type": "Completed",
//!     "name": "generate",
//!     "duration_ms": 1234,
//!     "idempotency_key": "exec-1:generate:0"
//!   }
//! }
//! ```
//!
//! `category + type` forms the event identity for storage and querying.
//!
//! # event_type_string()
//!
//! Returns `"category.variant"` dot-delimited strings for all 39 variants.
//! These strings are written to `Event::event_type` in the EventLog.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::execution::ExecutionId;
use crate::signal::Signal;

/// Current event schema version. Increment on breaking changes.
///
/// See spec Section 3.8 for the versioning strategy:
/// - New variants: no bump
/// - New optional fields: no bump (use `#[serde(default)]`)
/// - New required fields, removed fields, changed semantics: bump + migration
pub const EVENT_SCHEMA_VERSION: u32 = 1;

/// An event in the execution log. The storage format.
///
/// Events are immutable and append-only. They are the source of truth
/// for all execution state. The event log is a strictly ordered sequence
/// of events per execution (ordered by `sequence`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub execution_id: ExecutionId,
    /// Monotonically increasing sequence number within an execution.
    /// Assigned by the EventLog implementation on append.
    pub sequence: u64,
    /// Dot-delimited event type string for storage and querying.
    /// Examples: "execution.started", "generation.completed", "command.started"
    pub event_type: String,
    /// The event payload. Typed variants are serialized to this for storage.
    pub payload: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    /// Schema version for forward-compatible deserialization.
    /// Current version = 1. See spec Section 3.8 for versioning strategy.
    pub schema_version: u32,
}

/// The unified pipeline event type. Every activity pushes these onto the channel.
///
/// Outer enum uses adjacently-tagged serde (`#[serde(tag = "category", content = "event")]`).
/// Each inner enum uses internally-tagged serde (`#[serde(tag = "type")]`).
///
/// Activities push events onto the channel; the Reactor receives and dispatches them.
/// Every PipelineEvent is also recorded to the EventLog.
///
/// Many producers. One channel. One Reactor consuming and dispatching.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "category", content = "event")]
pub enum PipelineEvent {
    /// Pipeline lifecycle: started, completed, failed, cancelled, iteration, validation.
    Execution(ExecutionEvent),
    /// LLM streaming and lifecycle: chunk, started, completed, failed, timed out.
    Generation(GenerationEvent),
    /// Command execution: started, completed, failed, available.
    Command(CommandEvent),
    /// Activity lifecycle: started, completed, failed, retried, heartbeat.
    Activity(ActivityEvent),
    /// Hook evaluation results: evaluated, blocked.
    Hook(HookOutcome),
    /// Model/command selection, provider resolution, legacy route completed.
    Selection(SelectionEvent),
    /// Context assembly: system prompt, command formatting, sampling, messages, response.
    Context(ContextEvent),
    /// External signals: received for processing, logged for observability.
    Signal(SignalEvent),
    /// Child execution lifecycle: spawned, completed.
    Child(ChildEvent),
    /// Resource budget events: warning, exhausted.
    Budget(BudgetEvent),
}

// ── Category enums ─────────────────────────────────────────────────────────

/// Pipeline lifecycle events.
///
/// Variants: Started, Completed, Failed, Cancelled, IterationCompleted,
/// ValidationPassed, ValidationFailed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionEvent {
    Started {
        pipeline_name: String,
        tenant_id: String,
        request_id: String,
        parent_id: Option<String>,
        depth: u32,
        budget: BudgetSnapshot,
    },
    Completed {
        generation_calls: u32,
        commands_executed: u32,
        iterations: u32,
        duration_ms: u64,
    },
    Failed {
        error: String,
        partial_text: Option<String>,
    },
    Cancelled {
        reason: String,
        partial_text: Option<String>,
    },
    IterationCompleted {
        iteration: u32,
        commands_executed_this_iteration: u32,
    },
    ValidationPassed,
    ValidationFailed {
        reason: String,
    },
}

/// LLM generation events.
///
/// Variants: Chunk (wraps GeneratedEvent), Started, Completed, Failed, TimedOut.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GenerationEvent {
    Chunk(GeneratedEvent),
    Started {
        model: String,
        message_count: usize,
    },
    Completed {
        model: String,
        response_message: weft_core::WeftMessage,
        generated_events: Vec<GeneratedEvent>,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },
    Failed {
        model: String,
        error: String,
    },
    TimedOut {
        model: String,
        timeout_secs: u64,
        chunks_received: u32,
    },
}

/// Command execution events.
///
/// Variants: Started, Completed, Failed, Available.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CommandEvent {
    Started {
        invocation: weft_core::CommandInvocation,
    },
    Completed {
        name: String,
        result: weft_core::CommandResult,
    },
    Failed {
        name: String,
        error: String,
    },
    Available {
        commands: Vec<weft_core::CommandStub>,
    },
}

/// Activity lifecycle events.
///
/// Variants: Started, Completed, Failed, Retried, Heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActivityEvent {
    Started {
        name: String,
    },
    Completed {
        name: String,
        duration_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        idempotency_key: Option<String>,
    },
    Failed {
        name: String,
        error: String,
        retryable: bool,
    },
    Retried {
        name: String,
        attempt: u32,
        backoff_ms: u64,
        error: String,
    },
    Heartbeat {
        activity_name: String,
    },
}

/// Hook evaluation outcomes.
///
/// Named `HookOutcome` to avoid collision with `weft_core::HookEvent`.
///
/// Variants: Evaluated, Blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HookOutcome {
    Evaluated {
        hook_event: String,
        hook_name: String,
        decision: String,
        duration_ms: u64,
    },
    Blocked {
        hook_event: String,
        hook_name: String,
        reason: String,
    },
}

/// Model and command selection, provider resolution, and legacy routing events.
///
/// Variants: ModelSelected, CommandsSelected, ProviderResolved, RouteCompleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SelectionEvent {
    ModelSelected {
        model_name: String,
        score: f32,
        all_scores: Vec<(String, f32)>,
    },
    CommandsSelected {
        selected: Vec<weft_core::CommandStub>,
        candidates_scored: usize,
    },
    ProviderResolved {
        model_name: String,
        model_id: String,
        provider_name: String,
        capabilities: Vec<String>,
        max_tokens: u32,
    },
    RouteCompleted {
        domain: String,
        routing: weft_core::RoutingActivity,
    },
}

/// Context assembly events.
///
/// Variants: SystemPromptAssembled, CommandsFormatted, SamplingUpdated,
/// PromptAssembled (deprecated), MessageInjected, ResponseAssembled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContextEvent {
    SystemPromptAssembled {
        prompt_length: usize,
        layer_count: u32,
        message_count: usize,
    },
    CommandsFormatted {
        format: CommandFormat,
        command_count: usize,
    },
    SamplingUpdated {
        max_tokens: u32,
        temperature: Option<f32>,
        top_p: Option<f32>,
    },
    /// Deprecated: replaced by SystemPromptAssembled. Retained for backward
    /// compatibility with serialized event logs.
    PromptAssembled {
        message_count: usize,
    },
    MessageInjected {
        message: weft_core::WeftMessage,
        source: MessageInjectionSource,
    },
    ResponseAssembled {
        response: weft_core::WeftResponse,
    },
}

/// External signal events.
///
/// Variants: Received (for processing by the dispatch loop),
/// Logged (for observability only — signal_type + payload).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SignalEvent {
    /// A signal received for processing by the dispatch loop.
    Received(Signal),
    /// A signal logged for observability (type + raw payload).
    Logged {
        signal_type: String,
        payload: serde_json::Value,
    },
}

/// Child execution lifecycle events.
///
/// Variants: Spawned, Completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ChildEvent {
    Spawned {
        child_id: String,
        pipeline_name: String,
        reason: String,
    },
    Completed {
        child_id: ExecutionId,
        status: String,
        result_summary: serde_json::Value,
    },
}

/// Resource budget events.
///
/// Variants: Warning, Exhausted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BudgetEvent {
    Warning { resource: String, remaining: u32 },
    Exhausted { resource: String },
}

// ── event_type_string ───────────────────────────────────────────────────────

impl PipelineEvent {
    /// Returns the dot-delimited event type string for storage and querying.
    ///
    /// Format: `"category.variant"` for all 39 variants.
    /// Used to populate `Event::event_type` in the EventLog.
    pub fn event_type_string(&self) -> &'static str {
        match self {
            PipelineEvent::Execution(e) => match e {
                ExecutionEvent::Started { .. } => "execution.started",
                ExecutionEvent::Completed { .. } => "execution.completed",
                ExecutionEvent::Failed { .. } => "execution.failed",
                ExecutionEvent::Cancelled { .. } => "execution.cancelled",
                ExecutionEvent::IterationCompleted { .. } => "execution.iteration_completed",
                ExecutionEvent::ValidationPassed => "execution.validation_passed",
                ExecutionEvent::ValidationFailed { .. } => "execution.validation_failed",
            },
            PipelineEvent::Generation(e) => match e {
                GenerationEvent::Chunk(_) => "generation.chunk",
                GenerationEvent::Started { .. } => "generation.started",
                GenerationEvent::Completed { .. } => "generation.completed",
                GenerationEvent::Failed { .. } => "generation.failed",
                GenerationEvent::TimedOut { .. } => "generation.timed_out",
            },
            PipelineEvent::Command(e) => match e {
                CommandEvent::Started { .. } => "command.started",
                CommandEvent::Completed { .. } => "command.completed",
                CommandEvent::Failed { .. } => "command.failed",
                CommandEvent::Available { .. } => "command.available",
            },
            PipelineEvent::Activity(e) => match e {
                ActivityEvent::Started { .. } => "activity.started",
                ActivityEvent::Completed { .. } => "activity.completed",
                ActivityEvent::Failed { .. } => "activity.failed",
                ActivityEvent::Retried { .. } => "activity.retried",
                ActivityEvent::Heartbeat { .. } => "activity.heartbeat",
            },
            PipelineEvent::Hook(e) => match e {
                HookOutcome::Evaluated { .. } => "hook.evaluated",
                HookOutcome::Blocked { .. } => "hook.blocked",
            },
            PipelineEvent::Selection(e) => match e {
                SelectionEvent::ModelSelected { .. } => "selection.model_selected",
                SelectionEvent::CommandsSelected { .. } => "selection.commands_selected",
                SelectionEvent::ProviderResolved { .. } => "selection.provider_resolved",
                SelectionEvent::RouteCompleted { .. } => "selection.route_completed",
            },
            PipelineEvent::Context(e) => match e {
                ContextEvent::SystemPromptAssembled { .. } => "context.system_prompt_assembled",
                ContextEvent::CommandsFormatted { .. } => "context.commands_formatted",
                ContextEvent::SamplingUpdated { .. } => "context.sampling_updated",
                ContextEvent::PromptAssembled { .. } => "context.prompt_assembled",
                ContextEvent::MessageInjected { .. } => "context.message_injected",
                ContextEvent::ResponseAssembled { .. } => "context.response_assembled",
            },
            PipelineEvent::Signal(e) => match e {
                SignalEvent::Received(_) => "signal.received",
                SignalEvent::Logged { .. } => "signal.logged",
            },
            PipelineEvent::Child(e) => match e {
                ChildEvent::Spawned { .. } => "child.spawned",
                ChildEvent::Completed { .. } => "child.completed",
            },
            PipelineEvent::Budget(e) => match e {
                BudgetEvent::Warning { .. } => "budget.warning",
                BudgetEvent::Exhausted { .. } => "budget.exhausted",
            },
        }
    }
}

// ── Supporting types ────────────────────────────────────────────────────────

/// Events produced by the generative source (LLM, code generator, etc.).
///
/// Named GeneratedEvent because the source is agnostic -- the Activity
/// trait allows any generative system. The generate activity streams
/// these onto the channel as they arrive from the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratedEvent {
    /// Content from the generative source, as a Weft Wire ContentPart.
    Content { part: weft_core::ContentPart },
    /// A command invocation found in the generated output.
    CommandInvocation(weft_core::CommandInvocation),
    /// The generative source is done. No more content, no more commands.
    Done,
    /// Reasoning/thinking tokens. Not part of the user-visible response.
    Reasoning { content: String },
    /// The generative source refused the request.
    Refused { reason: String },
}

/// Snapshot of budget state for inclusion in events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub max_generation_calls: u32,
    pub remaining_generation_calls: u32,
    pub max_iterations: u32,
    pub remaining_iterations: u32,
    pub max_depth: u32,
    pub current_depth: u32,
    pub deadline_epoch_ms: i64,
}

/// Why a message was injected into the conversation context.
///
/// Carried by `ContextEvent::MessageInjected` to explain the origin of the
/// injected message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageInjectionSource {
    /// Result from a completed command execution.
    CommandResult { command_name: String },
    /// Error from a failed command execution.
    CommandError { command_name: String },
    /// Feedback from a hook that blocked generation.
    HookFeedback { hook_name: String },
    /// Context injected by a `Signal::InjectContext`.
    SignalInjection,
    /// System prompt assembled by `SystemPromptAssemblyActivity`.
    SystemPromptAssembly,
    /// Command descriptions injected into the prompt for providers without native tool support.
    CommandFormatInjection,
}

/// How commands were presented to the provider.
///
/// Carried by `ContextEvent::CommandsFormatted`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommandFormat {
    /// Native structured commands (provider supports tool_calling).
    Structured,
    /// Commands injected as text in the system prompt.
    PromptInjected,
    /// No commands (empty selection or commands not needed).
    NoCommands,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    fn make_message() -> weft_core::WeftMessage {
        weft_core::WeftMessage {
            role: weft_core::Role::Assistant,
            source: weft_core::Source::Provider,
            model: Some("test-model".to_string()),
            content: vec![weft_core::ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        }
    }

    fn make_snapshot() -> BudgetSnapshot {
        BudgetSnapshot {
            max_generation_calls: 10,
            remaining_generation_calls: 10,
            max_iterations: 5,
            remaining_iterations: 5,
            max_depth: 3,
            current_depth: 0,
            deadline_epoch_ms: 9_999_999_999,
        }
    }

    // ── Schema version ────────────────────────────────────────────────────

    #[test]
    fn event_schema_version_is_one() {
        assert_eq!(EVENT_SCHEMA_VERSION, 1);
    }

    // ── Serde round-trip: adjacently-tagged format ─────────────────────────
    //
    // Each case verifies that a PipelineEvent serializes to the adjacently-tagged
    // format {"category":"...","event":{"type":"...",...}} and that the
    // event_type_string() returns the expected dot-delimited string.

    #[rstest]
    // Execution (7 variants)
    #[case::execution_started(
        PipelineEvent::Execution(ExecutionEvent::Started {
            pipeline_name: "default".to_string(),
            tenant_id: "t1".to_string(),
            request_id: "r1".to_string(),
            parent_id: None,
            depth: 0,
            budget: BudgetSnapshot {
                max_generation_calls: 10,
                remaining_generation_calls: 10,
                max_iterations: 5,
                remaining_iterations: 5,
                max_depth: 3,
                current_depth: 0,
                deadline_epoch_ms: 0,
            },
        }),
        "execution.started",
        "Execution",
        "Started",
    )]
    #[case::execution_completed(
        PipelineEvent::Execution(ExecutionEvent::Completed {
            generation_calls: 1,
            commands_executed: 0,
            iterations: 1,
            duration_ms: 500,
        }),
        "execution.completed",
        "Execution",
        "Completed",
    )]
    #[case::execution_failed(
        PipelineEvent::Execution(ExecutionEvent::Failed { error: "err".to_string(), partial_text: None }),
        "execution.failed",
        "Execution",
        "Failed",
    )]
    #[case::execution_cancelled(
        PipelineEvent::Execution(ExecutionEvent::Cancelled { reason: "user".to_string(), partial_text: None }),
        "execution.cancelled",
        "Execution",
        "Cancelled",
    )]
    #[case::execution_iteration_completed(
        PipelineEvent::Execution(ExecutionEvent::IterationCompleted {
            iteration: 1,
            commands_executed_this_iteration: 2,
        }),
        "execution.iteration_completed",
        "Execution",
        "IterationCompleted",
    )]
    #[case::execution_validation_passed(
        PipelineEvent::Execution(ExecutionEvent::ValidationPassed),
        "execution.validation_passed",
        "Execution",
        "ValidationPassed"
    )]
    #[case::execution_validation_failed(
        PipelineEvent::Execution(ExecutionEvent::ValidationFailed { reason: "empty messages".to_string() }),
        "execution.validation_failed",
        "Execution",
        "ValidationFailed",
    )]
    // Generation (5 variants)
    #[case::generation_chunk(
        PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Done)),
        "generation.chunk",
        "Generation",
        "Chunk"
    )]
    #[case::generation_started(
        PipelineEvent::Generation(GenerationEvent::Started { model: "gpt-4".to_string(), message_count: 3 }),
        "generation.started",
        "Generation",
        "Started",
    )]
    #[case::generation_completed(
        PipelineEvent::Generation(GenerationEvent::Completed {
            model: "gpt-4".to_string(),
            response_message: weft_core::WeftMessage {
                role: weft_core::Role::Assistant,
                source: weft_core::Source::Provider,
                model: Some("gpt-4".to_string()),
                content: vec![],
                delta: false,
                message_index: 0,
            },
            generated_events: vec![],
            input_tokens: Some(100),
            output_tokens: Some(50),
        }),
        "generation.completed",
        "Generation",
        "Completed",
    )]
    #[case::generation_failed(
        PipelineEvent::Generation(GenerationEvent::Failed { model: "gpt-4".to_string(), error: "fail".to_string() }),
        "generation.failed",
        "Generation",
        "Failed",
    )]
    #[case::generation_timed_out(
        PipelineEvent::Generation(GenerationEvent::TimedOut {
            model: "gpt-4".to_string(),
            timeout_secs: 30,
            chunks_received: 5,
        }),
        "generation.timed_out",
        "Generation",
        "TimedOut",
    )]
    // Command (4 variants)
    #[case::command_started(
        PipelineEvent::Command(CommandEvent::Started {
            invocation: weft_core::CommandInvocation {
                name: "search".to_string(),
                action: weft_core::CommandAction::Execute,
                arguments: serde_json::json!({}),
            },
        }),
        "command.started",
        "Command",
        "Started",
    )]
    #[case::command_completed(
        PipelineEvent::Command(CommandEvent::Completed {
            name: "search".to_string(),
            result: weft_core::CommandResult {
                command_name: "search".to_string(),
                success: true,
                output: "ok".to_string(),
                error: None,
            },
        }),
        "command.completed",
        "Command",
        "Completed",
    )]
    #[case::command_failed(
        PipelineEvent::Command(CommandEvent::Failed { name: "search".to_string(), error: "err".to_string() }),
        "command.failed",
        "Command",
        "Failed",
    )]
    #[case::command_available(
        PipelineEvent::Command(CommandEvent::Available { commands: vec![] }),
        "command.available",
        "Command",
        "Available",
    )]
    // Activity (5 variants)
    #[case::activity_started(
        PipelineEvent::Activity(ActivityEvent::Started { name: "generate".to_string() }),
        "activity.started",
        "Activity",
        "Started",
    )]
    #[case::activity_completed(
        PipelineEvent::Activity(ActivityEvent::Completed {
            name: "generate".to_string(),
            duration_ms: 200,
            idempotency_key: Some("exec-1:generate:0".to_string()),
        }),
        "activity.completed",
        "Activity",
        "Completed",
    )]
    #[case::activity_failed(
        PipelineEvent::Activity(ActivityEvent::Failed {
            name: "generate".to_string(),
            error: "timeout".to_string(),
            retryable: true,
        }),
        "activity.failed",
        "Activity",
        "Failed",
    )]
    #[case::activity_retried(
        PipelineEvent::Activity(ActivityEvent::Retried {
            name: "generate".to_string(),
            attempt: 1,
            backoff_ms: 500,
            error: "transient".to_string(),
        }),
        "activity.retried",
        "Activity",
        "Retried",
    )]
    #[case::activity_heartbeat(
        PipelineEvent::Activity(ActivityEvent::Heartbeat { activity_name: "generate".to_string() }),
        "activity.heartbeat",
        "Activity",
        "Heartbeat",
    )]
    // Hook (2 variants)
    #[case::hook_evaluated(
        PipelineEvent::Hook(HookOutcome::Evaluated {
            hook_event: "pre_request".to_string(),
            hook_name: "auth".to_string(),
            decision: "allow".to_string(),
            duration_ms: 10,
        }),
        "hook.evaluated",
        "Hook",
        "Evaluated",
    )]
    #[case::hook_blocked(
        PipelineEvent::Hook(HookOutcome::Blocked {
            hook_event: "pre_request".to_string(),
            hook_name: "auth".to_string(),
            reason: "policy".to_string(),
        }),
        "hook.blocked",
        "Hook",
        "Blocked",
    )]
    // Selection (4 variants)
    #[case::selection_model_selected(
        PipelineEvent::Selection(SelectionEvent::ModelSelected {
            model_name: "gpt-4".to_string(),
            score: 0.9,
            all_scores: vec![("gpt-4".to_string(), 0.9)],
        }),
        "selection.model_selected",
        "Selection",
        "ModelSelected",
    )]
    #[case::selection_commands_selected(
        PipelineEvent::Selection(SelectionEvent::CommandsSelected { selected: vec![], candidates_scored: 0 }),
        "selection.commands_selected",
        "Selection",
        "CommandsSelected",
    )]
    #[case::selection_provider_resolved(
        PipelineEvent::Selection(SelectionEvent::ProviderResolved {
            model_name: "gpt-4".to_string(),
            model_id: "gpt-4-v1".to_string(),
            provider_name: "openai".to_string(),
            capabilities: vec!["chat_completions".to_string()],
            max_tokens: 4096,
        }),
        "selection.provider_resolved",
        "Selection",
        "ProviderResolved",
    )]
    #[case::selection_route_completed(
        PipelineEvent::Selection(SelectionEvent::RouteCompleted {
            domain: "model".to_string(),
            routing: weft_core::RoutingActivity {
                model: "gpt-4".to_string(),
                score: 0.9,
                filters: vec![],
            },
        }),
        "selection.route_completed",
        "Selection",
        "RouteCompleted",
    )]
    // Context (6 variants)
    #[case::context_system_prompt_assembled(
        PipelineEvent::Context(ContextEvent::SystemPromptAssembled {
            prompt_length: 100,
            layer_count: 1,
            message_count: 3,
        }),
        "context.system_prompt_assembled",
        "Context",
        "SystemPromptAssembled",
    )]
    #[case::context_commands_formatted(
        PipelineEvent::Context(ContextEvent::CommandsFormatted { format: CommandFormat::Structured, command_count: 2 }),
        "context.commands_formatted",
        "Context",
        "CommandsFormatted",
    )]
    #[case::context_sampling_updated(
        PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens: 4096, temperature: None, top_p: None }),
        "context.sampling_updated",
        "Context",
        "SamplingUpdated",
    )]
    #[case::context_prompt_assembled(
        PipelineEvent::Context(ContextEvent::PromptAssembled { message_count: 5 }),
        "context.prompt_assembled",
        "Context",
        "PromptAssembled",
    )]
    #[case::context_message_injected(
        PipelineEvent::Context(ContextEvent::MessageInjected {
            message: weft_core::WeftMessage {
                role: weft_core::Role::Assistant,
                source: weft_core::Source::Provider,
                model: Some("test".to_string()),
                content: vec![],
                delta: false,
                message_index: 0,
            },
            source: MessageInjectionSource::SignalInjection,
        }),
        "context.message_injected",
        "Context",
        "MessageInjected",
    )]
    #[case::context_response_assembled(
        PipelineEvent::Context(ContextEvent::ResponseAssembled {
            response: weft_core::WeftResponse {
                id: "r1".to_string(),
                model: "gpt-4".to_string(),
                messages: vec![],
                usage: weft_core::WeftUsage::default(),
                timing: weft_core::WeftTiming::default(),
            },
        }),
        "context.response_assembled",
        "Context",
        "ResponseAssembled",
    )]
    // Signal (2 variants)
    #[case::signal_received(
        PipelineEvent::Signal(SignalEvent::Received(Signal::Pause)),
        "signal.received",
        "Signal",
        "Received"
    )]
    #[case::signal_logged(
        PipelineEvent::Signal(SignalEvent::Logged {
            signal_type: "cancel".to_string(),
            payload: serde_json::Value::Null,
        }),
        "signal.logged",
        "Signal",
        "Logged",
    )]
    // Child (2 variants)
    #[case::child_spawned(
        PipelineEvent::Child(ChildEvent::Spawned {
            child_id: "child-1".to_string(),
            pipeline_name: "sub".to_string(),
            reason: "tool call".to_string(),
        }),
        "child.spawned",
        "Child",
        "Spawned",
    )]
    #[case::child_completed(
        PipelineEvent::Child(ChildEvent::Completed {
            child_id: ExecutionId::default(),
            status: "completed".to_string(),
            result_summary: serde_json::Value::Null,
        }),
        "child.completed",
        "Child",
        "Completed",
    )]
    // Budget (2 variants)
    #[case::budget_warning(
        PipelineEvent::Budget(BudgetEvent::Warning { resource: "gen".to_string(), remaining: 2 }),
        "budget.warning",
        "Budget",
        "Warning",
    )]
    #[case::budget_exhausted(
        PipelineEvent::Budget(BudgetEvent::Exhausted { resource: "gen".to_string() }),
        "budget.exhausted",
        "Budget",
        "Exhausted",
    )]
    fn pipeline_event_serde_round_trip_and_type_string(
        #[case] event: PipelineEvent,
        #[case] expected_type: &str,
        #[case] expected_category: &str,
        #[case] expected_inner_type: &str,
    ) {
        // Verify event_type_string()
        assert_eq!(event.event_type_string(), expected_type);

        // Serde round-trip
        let json = serde_json::to_string(&event).expect("serialization must succeed");
        let back: PipelineEvent =
            serde_json::from_str(&json).expect("deserialization must succeed");
        assert_eq!(back.event_type_string(), expected_type);

        // Verify adjacently-tagged format: must contain "category":"..." and "type":"..."
        let v: serde_json::Value = serde_json::from_str(&json).expect("must parse as JSON");
        assert_eq!(
            v["category"].as_str().unwrap_or(""),
            expected_category,
            "category field mismatch"
        );
        assert_eq!(
            v["event"]["type"].as_str().unwrap_or(""),
            expected_inner_type,
            "inner type field mismatch"
        );
    }

    // ── Adjacently-tagged format verification ──────────────────────────────

    #[test]
    fn activity_completed_serde_format() {
        let event = PipelineEvent::Activity(ActivityEvent::Completed {
            name: "generate".to_string(),
            duration_ms: 1234,
            idempotency_key: Some("exec-1:generate:0".to_string()),
        });
        let json = serde_json::to_string(&event).expect("must serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("must parse");
        assert_eq!(v["category"], "Activity");
        assert_eq!(v["event"]["type"], "Completed");
        assert_eq!(v["event"]["name"], "generate");
        assert_eq!(v["event"]["duration_ms"], 1234);
        assert_eq!(v["event"]["idempotency_key"], "exec-1:generate:0");
    }

    #[test]
    fn activity_completed_idempotency_key_absent_when_none() {
        let event = PipelineEvent::Activity(ActivityEvent::Completed {
            name: "validate".to_string(),
            duration_ms: 10,
            idempotency_key: None,
        });
        let json = serde_json::to_string(&event).expect("must serialize");
        // idempotency_key must not appear in JSON when None
        assert!(!json.contains("idempotency_key"));
    }

    // ── GeneratedEvent serializes ──────────────────────────────────────────

    #[test]
    fn generated_event_all_variants_serialize() {
        let variants = vec![
            GeneratedEvent::Done,
            GeneratedEvent::Content {
                part: weft_core::ContentPart::Text("hello".to_string()),
            },
            GeneratedEvent::Reasoning {
                content: "thinking...".to_string(),
            },
            GeneratedEvent::Refused {
                reason: "content policy".to_string(),
            },
        ];
        for v in &variants {
            let json = serde_json::to_string(v).expect("must serialize");
            let _back: GeneratedEvent = serde_json::from_str(&json).expect("must deserialize");
        }
    }

    // ── BudgetSnapshot serde ───────────────────────────────────────────────

    #[test]
    fn budget_snapshot_serde_round_trip() {
        let snap = make_snapshot();
        let json = serde_json::to_string(&snap).expect("must serialize");
        let back: BudgetSnapshot = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(back.remaining_generation_calls, 10);
        assert_eq!(back.current_depth, 0);
    }

    // ── Event struct serde ─────────────────────────────────────────────────

    #[test]
    fn event_struct_preserves_schema_version() {
        let exec_id = ExecutionId::new();
        let event = Event {
            execution_id: exec_id.clone(),
            sequence: 7,
            event_type: "activity.completed".to_string(),
            payload: serde_json::json!({ "name": "generate" }),
            timestamp: Utc::now(),
            schema_version: EVENT_SCHEMA_VERSION,
        };
        let json = serde_json::to_string(&event).expect("must serialize");
        assert!(json.contains("schema_version"));
        let back: Event = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(back.schema_version, 1);
        assert_eq!(back.sequence, 7);
        assert_eq!(back.execution_id, exec_id);
    }

    // ── ServiceLocator and ChildSpawner object-safety ─────────────────────
    // Verified here since they live in service.rs but PipelineEvent is the primary
    // type they interact with. Actual service.rs has its own tests.

    // ── MessageInjectionSource round-trips ────────────────────────────────

    #[test]
    fn message_injection_source_all_variants_serialize() {
        let sources = vec![
            MessageInjectionSource::CommandResult {
                command_name: "cmd".to_string(),
            },
            MessageInjectionSource::CommandError {
                command_name: "cmd".to_string(),
            },
            MessageInjectionSource::HookFeedback {
                hook_name: "hook".to_string(),
            },
            MessageInjectionSource::SignalInjection,
            MessageInjectionSource::SystemPromptAssembly,
            MessageInjectionSource::CommandFormatInjection,
        ];
        for source in &sources {
            let json = serde_json::to_string(source).expect("must serialize");
            let _back: MessageInjectionSource =
                serde_json::from_str(&json).expect("must deserialize");
        }
    }

    #[test]
    fn message_injected_command_result_preserves_name() {
        let msg = make_message();
        let event = PipelineEvent::Context(ContextEvent::MessageInjected {
            message: msg.clone(),
            source: MessageInjectionSource::CommandResult {
                command_name: "web_search".to_string(),
            },
        });
        let json = serde_json::to_string(&event).expect("must serialize");
        let back: PipelineEvent = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(back.event_type_string(), "context.message_injected");
        match back {
            PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::CommandResult { command_name },
            }) => {
                assert_eq!(command_name, "web_search");
                assert_eq!(message.model, msg.model);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ── CommandFormat variants ─────────────────────────────────────────────

    #[rstest]
    #[case::structured(CommandFormat::Structured, 3)]
    #[case::prompt_injected(CommandFormat::PromptInjected, 2)]
    #[case::no_commands(CommandFormat::NoCommands, 0)]
    fn commands_formatted_preserves_format_and_count(
        #[case] format: CommandFormat,
        #[case] count: usize,
    ) {
        let event = PipelineEvent::Context(ContextEvent::CommandsFormatted {
            format: format.clone(),
            command_count: count,
        });
        let json = serde_json::to_string(&event).expect("must serialize");
        let back: PipelineEvent = serde_json::from_str(&json).expect("must deserialize");
        match back {
            PipelineEvent::Context(ContextEvent::CommandsFormatted {
                format: back_format,
                command_count,
            }) => {
                assert_eq!(back_format, format);
                assert_eq!(command_count, count);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ── Generation token counts preserved ─────────────────────────────────

    #[test]
    fn generation_completed_token_counts_preserved() {
        let event = PipelineEvent::Generation(GenerationEvent::Completed {
            model: "gpt-4".to_string(),
            response_message: make_message(),
            generated_events: vec![],
            input_tokens: Some(100),
            output_tokens: Some(50),
        });
        let json = serde_json::to_string(&event).expect("must serialize");
        let back: PipelineEvent = serde_json::from_str(&json).expect("must deserialize");
        match back {
            PipelineEvent::Generation(GenerationEvent::Completed {
                input_tokens,
                output_tokens,
                ..
            }) => {
                assert_eq!(input_tokens, Some(100));
                assert_eq!(output_tokens, Some(50));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ── Nested pattern: Generation(Chunk(Content)) ─────────────────────────

    #[test]
    fn generation_chunk_content_three_level_pattern() {
        let event = PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content {
            part: weft_core::ContentPart::Text("hello".to_string()),
        }));
        let json = serde_json::to_string(&event).expect("must serialize");
        let back: PipelineEvent = serde_json::from_str(&json).expect("must deserialize");
        // Three-level nested pattern from spec Section 10.9
        if let PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content { part })) =
            &back
        {
            assert!(matches!(part, weft_core::ContentPart::Text(_)));
        } else {
            panic!("three-level pattern failed to match: {back:?}");
        }
    }

    // ── ModelSelected scores ───────────────────────────────────────────────

    #[test]
    fn model_selected_preserves_scores() {
        let event = PipelineEvent::Selection(SelectionEvent::ModelSelected {
            model_name: "gpt-4-turbo".to_string(),
            score: 0.87,
            all_scores: vec![
                ("gpt-4-turbo".to_string(), 0.87),
                ("claude-3-opus".to_string(), 0.72),
            ],
        });
        let json = serde_json::to_string(&event).expect("must serialize");
        let back: PipelineEvent = serde_json::from_str(&json).expect("must deserialize");
        match back {
            PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name,
                score,
                all_scores,
            }) => {
                assert_eq!(model_name, "gpt-4-turbo");
                assert!((score - 0.87_f32).abs() < 1e-5);
                assert_eq!(all_scores.len(), 2);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
