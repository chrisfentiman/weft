//! Pipeline event types and the event storage format.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::execution::ExecutionId;

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
    /// Current version = 1. See Section 3.8 for versioning strategy.
    pub schema_version: u32,
}

/// The unified event type. Everything goes on one channel.
///
/// Activities push these onto the channel. The Reactor receives
/// and dispatches them. Every PipelineEvent is also recorded to
/// the EventLog.
///
/// Many producers. One channel. One Reactor consuming and dispatching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineEvent {
    // ── From generative source (streamed token by token) ────────
    Generated(GeneratedEvent),

    // ── From command execution ──────────────────────────────────
    CommandCompleted {
        name: String,
        result: weft_core::CommandResult,
    },
    CommandFailed {
        name: String,
        error: String,
    },

    // ── From external (user, API, webhook) ──────────────────────
    Signal(crate::signal::Signal),

    // ── From child executions ───────────────────────────────────
    ChildCompleted {
        child_id: ExecutionId,
        status: String,
        result_summary: serde_json::Value,
    },

    // ── From budget tracker ─────────────────────────────────────
    BudgetWarning {
        resource: String,
        remaining: u32,
    },
    BudgetExhausted {
        resource: String,
    },

    // ── From hook execution ─────────────────────────────────────
    HookEvaluated {
        hook_event: String,
        hook_name: String,
        decision: String,
        duration_ms: u64,
    },
    HookBlocked {
        hook_event: String,
        hook_name: String,
        reason: String,
    },

    // ── Activity lifecycle ──────────────────────────────────────
    ActivityStarted {
        name: String,
    },
    ActivityCompleted {
        name: String,
        duration_ms: u64,
        /// Idempotency key for deduplication on retry/replay.
        /// Present for generate and execute_command activities.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        idempotency_key: Option<String>,
    },
    ActivityFailed {
        name: String,
        error: String,
        /// Whether this failure is eligible for retry. Set by the
        /// activity. The Reactor checks this before applying RetryPolicy.
        retryable: bool,
    },

    // ── Activity retry lifecycle ─────────────────────────────────
    /// An activity is being retried after a transient failure.
    /// Records the attempt number (1-indexed: attempt 1 = first retry,
    /// not the initial attempt) and backoff duration before this attempt.
    ActivityRetried {
        name: String,
        attempt: u32,
        backoff_ms: u64,
        error: String,
    },

    // ── Generation timeout ───────────────────────────────────────
    /// The generate activity timed out waiting for a chunk from the
    /// provider. The timeout is per-chunk (resets on each received chunk),
    /// not per-generation-call. Follows the retry policy if configured.
    GenerationTimedOut {
        model: String,
        timeout_secs: u64,
        /// How many chunks were received before the timeout.
        chunks_received: u32,
    },

    // ── Heartbeat ────────────────────────────────────────────────
    /// Periodic liveness signal from a long-running activity.
    /// The Reactor tracks these per activity and cancels activities
    /// that stop heartbeating.
    Heartbeat {
        activity_name: String,
    },

    // ── Routing ─────────────────────────────────────────────────
    RouteCompleted {
        domain: String,
        routing: weft_core::RoutingActivity,
    },

    // ── Generation lifecycle ────────────────────────────────────
    GenerationStarted {
        model: String,
        message_count: usize,
    },
    GenerationCompleted {
        model: String,
        /// The full response as a Weft Wire message.
        response_message: weft_core::WeftMessage,
        /// Parsed events extracted from the response message.
        generated_events: Vec<GeneratedEvent>,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },
    GenerationFailed {
        model: String,
        error: String,
    },

    // ── Command lifecycle (detailed) ────────────────────────────
    CommandStarted {
        invocation: weft_core::CommandInvocation,
    },

    // ── Prompt assembly ─────────────────────────────────────────
    /// Deprecated: replaced by SystemPromptAssembled. Retained for backward
    /// compatibility with serialized event logs.
    PromptAssembled {
        message_count: usize,
    },

    // ── Execution lifecycle ─────────────────────────────────────
    ExecutionStarted {
        pipeline_name: String,
        tenant_id: String,
        request_id: String,
        parent_id: Option<String>,
        depth: u32,
        budget: BudgetSnapshot,
    },
    ExecutionCompleted {
        generation_calls: u32,
        commands_executed: u32,
        iterations: u32,
        duration_ms: u64,
    },
    ExecutionFailed {
        error: String,
        partial_text: Option<String>,
    },
    ExecutionCancelled {
        reason: String,
        partial_text: Option<String>,
    },

    // ── Iteration lifecycle ─────────────────────────────────────
    IterationCompleted {
        iteration: u32,
        commands_executed_this_iteration: u32,
    },

    // ── Signal receipt (for logging) ────────────────────────────
    SignalReceived {
        signal_type: String,
        payload: serde_json::Value,
    },

    // ── Child spawning ──────────────────────────────────────────
    ChildSpawned {
        child_id: String,
        pipeline_name: String,
        reason: String,
    },

    // ── Validation ──────────────────────────────────────────────
    ValidationPassed,
    ValidationFailed {
        reason: String,
    },

    // ── Response assembly ───────────────────────────────────────
    ResponseAssembled {
        response: weft_core::WeftResponse,
    },

    // ── Message injection ────────────────────────────────────────
    /// A message was added to the conversation context by the Reactor.
    ///
    /// Recorded every time the Reactor pushes a `WeftMessage` into `state.messages`
    /// (command results, hook feedback, signal-injected context). This makes the
    /// message list fully reconstructable from the event log: start with the original
    /// request messages, then append each `MessageInjected` payload in sequence order.
    MessageInjected {
        message: weft_core::WeftMessage,
        source: MessageInjectionSource,
    },

    // ── Commands availability ────────────────────────────────────
    /// The available commands for this execution, as determined by the validate
    /// activity querying the command registry.
    ///
    /// Pushed by `ValidateActivity` after loading commands from the registry.
    /// The Reactor uses this to populate `state.available_commands`, which is
    /// passed to subsequent activities via `ActivityInput`. Recording this event
    /// makes the available command set reconstructable from the event log.
    CommandsAvailable {
        commands: Vec<weft_core::CommandStub>,
    },

    // ── Selection (replaces RouteCompleted for model) ────────────────
    /// A model was selected via semantic routing.
    ///
    /// Emitted by `ModelSelectionActivity` after scoring all model candidates.
    /// Includes all scored candidates for observability and replay.
    ModelSelected {
        /// The routing name of the selected model (e.g., "gpt-4-turbo").
        model_name: String,
        /// Routing score for the selected model (0.0–1.0).
        score: f32,
        /// All candidates and their scores, for observability.
        all_scores: Vec<(String, f32)>,
    },

    /// Commands were selected via semantic routing.
    ///
    /// Emitted by `CommandSelectionActivity` after scoring and filtering.
    /// Includes the filtered set and the number of candidates scored.
    CommandsSelected {
        /// Commands selected after threshold + max_commands filtering.
        selected: Vec<weft_core::CommandStub>,
        /// Number of candidates scored (before filtering).
        candidates_scored: usize,
    },

    // ── Provider resolution ──────────────────────────────────────────
    /// Provider and model capabilities were resolved for the selected model.
    ///
    /// Emitted by `ProviderResolutionActivity`. Carries all resolved data so
    /// downstream activities can use it without re-querying the provider service.
    ProviderResolved {
        /// The model routing name (same as `ModelSelected.model_name`).
        model_name: String,
        /// The model API identifier sent to the provider (e.g., "claude-sonnet-4-20250514").
        model_id: String,
        /// Provider name from config (e.g., "anthropic").
        provider_name: String,
        /// Capabilities this model supports, as strings for serialization.
        /// Converted from `HashSet<Capability>` via `cap.as_str().to_string()`.
        /// e.g., `["chat_completions", "tool_calling"]`.
        capabilities: Vec<String>,
        /// Max tokens for this model.
        max_tokens: u32,
    },

    // ── System prompt ────────────────────────────────────────────────
    /// System prompt was assembled from gateway, agent, and caller layers.
    ///
    /// Emitted by `SystemPromptAssemblyActivity` after assembling the prompt.
    /// The actual prompt content is carried by the preceding `MessageInjected`
    /// event (source `SystemPromptAssembly`).
    SystemPromptAssembled {
        /// Length of the assembled system prompt in characters.
        prompt_length: usize,
        /// Number of layers that contributed (gateway, agent, caller).
        layer_count: u32,
        /// Total message count after system prompt insertion.
        message_count: usize,
    },

    // ── Command formatting ──────────────────────────────────────────
    /// Commands were formatted for the provider.
    ///
    /// Emitted by `CommandFormattingActivity`. The format indicates whether
    /// commands were passed as structured tool definitions or injected as text.
    CommandsFormatted {
        /// How commands were formatted.
        format: CommandFormat,
        /// Number of commands formatted.
        command_count: usize,
    },

    // ── Sampling update ─────────────────────────────────────────────
    /// Sampling parameters were adjusted for the selected model.
    ///
    /// Emitted by `SamplingAdjustmentActivity`. Records current values
    /// (after clamping) not the diff from originals.
    SamplingUpdated {
        /// Current max_tokens after clamping to model limit.
        max_tokens: u32,
        /// Current temperature. None means no temperature specified.
        temperature: Option<f32>,
        /// Current top_p. None means no top_p specified.
        top_p: Option<f32>,
    },
}

/// Why a message was injected into the conversation context.
///
/// Carried by `PipelineEvent::MessageInjected` to explain the origin of the
/// injected message. This allows replay code to correctly categorize injected
/// messages and observers to understand why the message list grew.
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
    ///
    /// The reactor inserts this message at `messages[0]`, replacing any
    /// existing system-role message. This ensures the provider always sees
    /// the gateway-assembled prompt as the canonical system prompt.
    SystemPromptAssembly,
    /// Command descriptions injected into the prompt for providers without native tool support.
    ///
    /// Used by `CommandFormattingActivity` when the model does not support
    /// `tool_calling`. The message is inserted after the system prompt.
    CommandFormatInjection,
}

/// How commands were presented to the provider.
///
/// Carried by `PipelineEvent::CommandsFormatted` to indicate whether
/// commands were passed as structured tool definitions (native provider
/// support) or injected as text in the system prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommandFormat {
    /// Native structured commands (provider supports tool_calling).
    Structured,
    /// Commands injected as text in the system prompt.
    PromptInjected,
    /// No commands (empty selection or commands not needed).
    NoCommands,
}

impl PipelineEvent {
    /// Returns the dot-delimited event type string for storage and querying.
    ///
    /// Used to populate `Event::event_type` in the EventLog.
    pub fn event_type_string(&self) -> &'static str {
        match self {
            PipelineEvent::Generated(_) => "generated",
            PipelineEvent::CommandCompleted { .. } => "command.completed",
            PipelineEvent::CommandFailed { .. } => "command.failed",
            PipelineEvent::Signal(_) => "signal",
            PipelineEvent::ChildCompleted { .. } => "child.completed",
            PipelineEvent::BudgetWarning { .. } => "budget.warning",
            PipelineEvent::BudgetExhausted { .. } => "budget.exhausted",
            PipelineEvent::HookEvaluated { .. } => "hook.evaluated",
            PipelineEvent::HookBlocked { .. } => "hook.blocked",
            PipelineEvent::ActivityStarted { .. } => "activity.started",
            PipelineEvent::ActivityCompleted { .. } => "activity.completed",
            PipelineEvent::ActivityFailed { .. } => "activity.failed",
            PipelineEvent::ActivityRetried { .. } => "activity.retried",
            PipelineEvent::GenerationTimedOut { .. } => "generation.timed_out",
            PipelineEvent::Heartbeat { .. } => "heartbeat",
            PipelineEvent::RouteCompleted { .. } => "route.completed",
            PipelineEvent::GenerationStarted { .. } => "generation.started",
            PipelineEvent::GenerationCompleted { .. } => "generation.completed",
            PipelineEvent::GenerationFailed { .. } => "generation.failed",
            PipelineEvent::CommandStarted { .. } => "command.started",
            PipelineEvent::PromptAssembled { .. } => "prompt.assembled",
            PipelineEvent::ExecutionStarted { .. } => "execution.started",
            PipelineEvent::ExecutionCompleted { .. } => "execution.completed",
            PipelineEvent::ExecutionFailed { .. } => "execution.failed",
            PipelineEvent::ExecutionCancelled { .. } => "execution.cancelled",
            PipelineEvent::IterationCompleted { .. } => "iteration.completed",
            PipelineEvent::SignalReceived { .. } => "signal.received",
            PipelineEvent::ChildSpawned { .. } => "child.spawned",
            PipelineEvent::ValidationPassed => "validation.passed",
            PipelineEvent::ValidationFailed { .. } => "validation.failed",
            PipelineEvent::ResponseAssembled { .. } => "response.assembled",
            PipelineEvent::MessageInjected { .. } => "message.injected",
            PipelineEvent::CommandsAvailable { .. } => "commands.available",
            PipelineEvent::ModelSelected { .. } => "model.selected",
            PipelineEvent::CommandsSelected { .. } => "commands.selected",
            PipelineEvent::ProviderResolved { .. } => "provider.resolved",
            PipelineEvent::SystemPromptAssembled { .. } => "system_prompt.assembled",
            PipelineEvent::CommandsFormatted { .. } => "commands.formatted",
            PipelineEvent::SamplingUpdated { .. } => "sampling.updated",
        }
    }
}

/// Events produced by the generative source (LLM, code generator, etc.).
///
/// Named GeneratedEvent because the source is agnostic -- the Activity
/// trait allows any generative system. The generate activity streams
/// these onto the channel as they arrive from the provider.
///
/// Uses Weft Wire types natively. `ContentPart` carries typed content
/// (text, media, command calls). The pipeline never sees OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratedEvent {
    /// Content from the generative source, as a Weft Wire ContentPart.
    /// Most commonly `ContentPart::Text`, but can be any content type
    /// the provider produces.
    ///
    /// Streamed token-by-token. Each Content event is one token or
    /// chunk pushed onto the channel as it arrives.
    Content { part: weft_core::ContentPart },

    /// A command invocation found in the generated output.
    /// Uses the Weft Wire `CommandInvocation` type directly.
    CommandInvocation(weft_core::CommandInvocation),

    /// The generative source is done. No more content, no more commands.
    Done,

    /// Reasoning/thinking tokens. Not part of the user-visible response.
    /// Provider-specific: LLMs emit chain-of-thought here. Other
    /// generative sources may not produce reasoning events.
    Reasoning { content: String },

    /// The generative source refused the request.
    /// Provider-specific: LLMs may refuse for content policy reasons.
    /// Other sources may not produce refusal events.
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

#[cfg(test)]
mod tests {
    use super::*;
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    fn make_message() -> WeftMessage {
        WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("test-model".to_string()),
            content: vec![ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        }
    }

    #[test]
    fn test_event_schema_version_is_one() {
        assert_eq!(EVENT_SCHEMA_VERSION, 1);
    }

    #[test]
    fn test_event_type_string_all_variants() {
        // Verify every variant produces a non-empty event_type string
        let exec_id = ExecutionId::new();
        let snapshot = BudgetSnapshot {
            max_generation_calls: 10,
            remaining_generation_calls: 10,
            max_iterations: 5,
            remaining_iterations: 5,
            max_depth: 3,
            current_depth: 0,
            deadline_epoch_ms: 9999999999,
        };

        let events: Vec<PipelineEvent> = vec![
            PipelineEvent::Generated(GeneratedEvent::Done),
            PipelineEvent::CommandCompleted {
                name: "cmd".to_string(),
                result: weft_core::CommandResult {
                    command_name: "cmd".to_string(),
                    success: true,
                    output: "ok".to_string(),
                    error: None,
                },
            },
            PipelineEvent::CommandFailed {
                name: "cmd".to_string(),
                error: "err".to_string(),
            },
            PipelineEvent::Signal(crate::signal::Signal::Pause),
            PipelineEvent::ChildCompleted {
                child_id: exec_id.clone(),
                status: "completed".to_string(),
                result_summary: serde_json::Value::Null,
            },
            PipelineEvent::BudgetWarning {
                resource: "gen".to_string(),
                remaining: 1,
            },
            PipelineEvent::BudgetExhausted {
                resource: "gen".to_string(),
            },
            PipelineEvent::HookEvaluated {
                hook_event: "pre_request".to_string(),
                hook_name: "hook".to_string(),
                decision: "allow".to_string(),
                duration_ms: 5,
            },
            PipelineEvent::HookBlocked {
                hook_event: "pre_request".to_string(),
                hook_name: "hook".to_string(),
                reason: "blocked".to_string(),
            },
            PipelineEvent::ActivityStarted {
                name: "generate".to_string(),
            },
            PipelineEvent::ActivityCompleted {
                name: "generate".to_string(),
                duration_ms: 100,
                idempotency_key: None,
            },
            PipelineEvent::ActivityFailed {
                name: "generate".to_string(),
                error: "err".to_string(),
                retryable: true,
            },
            PipelineEvent::ActivityRetried {
                name: "generate".to_string(),
                attempt: 1,
                backoff_ms: 1000,
                error: "err".to_string(),
            },
            PipelineEvent::GenerationTimedOut {
                model: "gpt-4".to_string(),
                timeout_secs: 30,
                chunks_received: 5,
            },
            PipelineEvent::Heartbeat {
                activity_name: "generate".to_string(),
            },
            PipelineEvent::RouteCompleted {
                domain: "model".to_string(),
                routing: weft_core::RoutingActivity {
                    model: "gpt-4".to_string(),
                    score: 0.9,
                    filters: vec![],
                },
            },
            PipelineEvent::GenerationStarted {
                model: "gpt-4".to_string(),
                message_count: 3,
            },
            PipelineEvent::GenerationCompleted {
                model: "gpt-4".to_string(),
                response_message: make_message(),
                generated_events: vec![],
                input_tokens: Some(100),
                output_tokens: Some(50),
            },
            PipelineEvent::GenerationFailed {
                model: "gpt-4".to_string(),
                error: "err".to_string(),
            },
            PipelineEvent::CommandStarted {
                invocation: weft_core::CommandInvocation {
                    name: "search".to_string(),
                    action: weft_core::CommandAction::Execute,
                    arguments: serde_json::json!({}),
                },
            },
            PipelineEvent::PromptAssembled { message_count: 5 },
            PipelineEvent::ExecutionStarted {
                pipeline_name: "default".to_string(),
                tenant_id: "t1".to_string(),
                request_id: "r1".to_string(),
                parent_id: None,
                depth: 0,
                budget: snapshot,
            },
            PipelineEvent::ExecutionCompleted {
                generation_calls: 1,
                commands_executed: 0,
                iterations: 1,
                duration_ms: 500,
            },
            PipelineEvent::ExecutionFailed {
                error: "err".to_string(),
                partial_text: None,
            },
            PipelineEvent::ExecutionCancelled {
                reason: "user".to_string(),
                partial_text: None,
            },
            PipelineEvent::IterationCompleted {
                iteration: 1,
                commands_executed_this_iteration: 2,
            },
            PipelineEvent::SignalReceived {
                signal_type: "cancel".to_string(),
                payload: serde_json::Value::Null,
            },
            PipelineEvent::ChildSpawned {
                child_id: "child-1".to_string(),
                pipeline_name: "sub".to_string(),
                reason: "tool call".to_string(),
            },
            PipelineEvent::ValidationPassed,
            PipelineEvent::ValidationFailed {
                reason: "invalid".to_string(),
            },
            PipelineEvent::ResponseAssembled {
                response: weft_core::WeftResponse {
                    id: "r1".to_string(),
                    model: "gpt-4".to_string(),
                    messages: vec![],
                    usage: weft_core::WeftUsage::default(),
                    timing: weft_core::WeftTiming::default(),
                },
            },
            PipelineEvent::MessageInjected {
                message: make_message(),
                source: crate::event::MessageInjectionSource::CommandResult {
                    command_name: "search".to_string(),
                },
            },
            PipelineEvent::CommandsAvailable {
                commands: vec![weft_core::CommandStub {
                    name: "search".to_string(),
                    description: "Search the web".to_string(),
                }],
            },
        ];

        for event in &events {
            let type_str = event.event_type_string();
            assert!(
                !type_str.is_empty(),
                "event_type_string should not be empty"
            );
        }

        // Spot-check specific values
        assert_eq!(
            PipelineEvent::ValidationPassed.event_type_string(),
            "validation.passed"
        );
        assert_eq!(
            PipelineEvent::ExecutionStarted {
                pipeline_name: "x".to_string(),
                tenant_id: "t".to_string(),
                request_id: "r".to_string(),
                parent_id: None,
                depth: 0,
                budget: BudgetSnapshot {
                    max_generation_calls: 1,
                    remaining_generation_calls: 1,
                    max_iterations: 1,
                    remaining_iterations: 1,
                    max_depth: 1,
                    current_depth: 0,
                    deadline_epoch_ms: 0,
                },
            }
            .event_type_string(),
            "execution.started"
        );
    }

    #[test]
    fn test_pipeline_event_serde_round_trip() {
        let event = PipelineEvent::ActivityCompleted {
            name: "generate".to_string(),
            duration_ms: 200,
            idempotency_key: Some("exec-1:generate:0".to_string()),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "activity.completed");
    }

    #[test]
    fn test_activity_completed_idempotency_key_skipped_when_none() {
        let event = PipelineEvent::ActivityCompleted {
            name: "validate".to_string(),
            duration_ms: 10,
            idempotency_key: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        // idempotency_key should not appear in JSON when None
        assert!(!json.contains("idempotency_key"));
    }

    #[test]
    fn test_activity_failed_retryable_field() {
        let retryable = PipelineEvent::ActivityFailed {
            name: "generate".to_string(),
            error: "rate limit".to_string(),
            retryable: true,
        };
        let json = serde_json::to_string(&retryable).unwrap();
        assert!(json.contains("true"));

        let non_retryable = PipelineEvent::ActivityFailed {
            name: "generate".to_string(),
            error: "invalid input".to_string(),
            retryable: false,
        };
        let json2 = serde_json::to_string(&non_retryable).unwrap();
        assert!(json2.contains("false"));
    }

    #[test]
    fn test_generated_event_variants_serialize() {
        let variants = vec![
            GeneratedEvent::Done,
            GeneratedEvent::Content {
                part: ContentPart::Text("hello".to_string()),
            },
            GeneratedEvent::Reasoning {
                content: "thinking...".to_string(),
            },
            GeneratedEvent::Refused {
                reason: "content policy".to_string(),
            },
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let _back: GeneratedEvent = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_budget_snapshot_serde() {
        let snap = BudgetSnapshot {
            max_generation_calls: 10,
            remaining_generation_calls: 8,
            max_iterations: 5,
            remaining_iterations: 4,
            max_depth: 3,
            current_depth: 1,
            deadline_epoch_ms: 1_700_000_000_000,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: BudgetSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.remaining_generation_calls, 8);
        assert_eq!(back.current_depth, 1);
    }

    #[test]
    fn test_pipeline_event_activity_retried_round_trip() {
        let event = PipelineEvent::ActivityRetried {
            name: "generate".to_string(),
            attempt: 2,
            backoff_ms: 2000,
            error: "connection reset".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "activity.retried");
        match back {
            PipelineEvent::ActivityRetried {
                name,
                attempt,
                backoff_ms,
                error,
            } => {
                assert_eq!(name, "generate");
                assert_eq!(attempt, 2);
                assert_eq!(backoff_ms, 2000);
                assert_eq!(error, "connection reset");
            }
            other => panic!("expected ActivityRetried, got {:?}", other),
        }
    }

    #[test]
    fn test_pipeline_event_generation_timed_out_round_trip() {
        let event = PipelineEvent::GenerationTimedOut {
            model: "claude-3-opus".to_string(),
            timeout_secs: 60,
            chunks_received: 12,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "generation.timed_out");
        match back {
            PipelineEvent::GenerationTimedOut {
                model,
                timeout_secs,
                chunks_received,
            } => {
                assert_eq!(model, "claude-3-opus");
                assert_eq!(timeout_secs, 60);
                assert_eq!(chunks_received, 12);
            }
            other => panic!("expected GenerationTimedOut, got {:?}", other),
        }
    }

    #[test]
    fn test_pipeline_event_heartbeat_round_trip() {
        let event = PipelineEvent::Heartbeat {
            activity_name: "long-running-generate".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "heartbeat");
        match back {
            PipelineEvent::Heartbeat { activity_name } => {
                assert_eq!(activity_name, "long-running-generate");
            }
            other => panic!("expected Heartbeat, got {:?}", other),
        }
    }

    #[test]
    fn test_event_struct_serde_round_trip_preserves_schema_version() {
        let exec_id = ExecutionId::new();
        let event = Event {
            execution_id: exec_id.clone(),
            sequence: 7,
            event_type: "activity.retried".to_string(),
            payload: serde_json::json!({
                "name": "generate",
                "attempt": 1,
                "backoff_ms": 1000,
                "error": "timeout"
            }),
            timestamp: Utc::now(),
            schema_version: EVENT_SCHEMA_VERSION,
        };
        let json = serde_json::to_string(&event).unwrap();
        // schema_version must appear in the serialized output
        assert!(json.contains("schema_version"));
        let back: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, EVENT_SCHEMA_VERSION);
        assert_eq!(back.schema_version, 1);
        assert_eq!(back.sequence, 7);
        assert_eq!(back.event_type, "activity.retried");
        assert_eq!(back.execution_id, exec_id);
    }

    #[test]
    fn test_message_injected_command_result_round_trip() {
        let msg = make_message();
        let event = PipelineEvent::MessageInjected {
            message: msg.clone(),
            source: crate::event::MessageInjectionSource::CommandResult {
                command_name: "web_search".to_string(),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "message.injected");
        match back {
            PipelineEvent::MessageInjected {
                message,
                source: crate::event::MessageInjectionSource::CommandResult { command_name },
            } => {
                assert_eq!(command_name, "web_search");
                assert_eq!(message.model, msg.model);
                assert_eq!(message.role, msg.role);
            }
            other => panic!(
                "expected MessageInjected with CommandResult source, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_message_injected_signal_injection_round_trip() {
        let msg = make_message();
        let event = PipelineEvent::MessageInjected {
            message: msg.clone(),
            source: crate::event::MessageInjectionSource::SignalInjection,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "message.injected");
        match back {
            PipelineEvent::MessageInjected {
                message,
                source: crate::event::MessageInjectionSource::SignalInjection,
            } => {
                assert_eq!(message.model, msg.model);
            }
            other => panic!(
                "expected MessageInjected with SignalInjection source, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_message_injected_hook_feedback_round_trip() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text(
                "please try again without profanity".to_string(),
            )],
            delta: false,
            message_index: 1,
        };
        let event = PipelineEvent::MessageInjected {
            message: msg,
            source: crate::event::MessageInjectionSource::HookFeedback {
                hook_name: "content-policy-hook".to_string(),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "message.injected");
        match back {
            PipelineEvent::MessageInjected {
                source: crate::event::MessageInjectionSource::HookFeedback { hook_name },
                ..
            } => {
                assert_eq!(hook_name, "content-policy-hook");
            }
            other => panic!(
                "expected MessageInjected with HookFeedback source, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_commands_available_round_trip() {
        let commands = vec![
            weft_core::CommandStub {
                name: "web_search".to_string(),
                description: "Search the web".to_string(),
            },
            weft_core::CommandStub {
                name: "calculator".to_string(),
                description: "Evaluate math expressions".to_string(),
            },
        ];
        let event = PipelineEvent::CommandsAvailable {
            commands: commands.clone(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.available");
        match back {
            PipelineEvent::CommandsAvailable {
                commands: back_cmds,
            } => {
                assert_eq!(back_cmds.len(), 2);
                assert_eq!(back_cmds[0].name, "web_search");
                assert_eq!(back_cmds[1].name, "calculator");
                assert_eq!(back_cmds[0].description, "Search the web");
            }
            other => panic!("expected CommandsAvailable, got {:?}", other),
        }
    }

    #[test]
    fn test_commands_available_empty_list_round_trip() {
        let event = PipelineEvent::CommandsAvailable { commands: vec![] };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.available");
        match back {
            PipelineEvent::CommandsAvailable { commands } => {
                assert!(commands.is_empty());
            }
            other => panic!("expected CommandsAvailable, got {:?}", other),
        }
    }

    #[test]
    fn test_event_type_string_new_variants() {
        let msg = make_message();
        assert_eq!(
            PipelineEvent::MessageInjected {
                message: msg,
                source: crate::event::MessageInjectionSource::SignalInjection,
            }
            .event_type_string(),
            "message.injected"
        );
        assert_eq!(
            PipelineEvent::CommandsAvailable { commands: vec![] }.event_type_string(),
            "commands.available"
        );
    }

    #[test]
    fn test_message_injection_source_all_variants_serialize() {
        // Verify all MessageInjectionSource variants serialize without error,
        // including the new Phase 1 variants.
        let sources = vec![
            crate::event::MessageInjectionSource::CommandResult {
                command_name: "cmd".to_string(),
            },
            crate::event::MessageInjectionSource::CommandError {
                command_name: "cmd".to_string(),
            },
            crate::event::MessageInjectionSource::HookFeedback {
                hook_name: "hook".to_string(),
            },
            crate::event::MessageInjectionSource::SignalInjection,
            crate::event::MessageInjectionSource::SystemPromptAssembly,
            crate::event::MessageInjectionSource::CommandFormatInjection,
        ];
        for source in &sources {
            let json = serde_json::to_string(source).unwrap();
            let _back: crate::event::MessageInjectionSource = serde_json::from_str(&json).unwrap();
        }
    }

    // ── Phase 1: New variant tests ─────────────────────────────────────────────

    #[test]
    fn test_model_selected_round_trip() {
        let event = PipelineEvent::ModelSelected {
            model_name: "gpt-4-turbo".to_string(),
            score: 0.87,
            all_scores: vec![
                ("gpt-4-turbo".to_string(), 0.87),
                ("claude-3-opus".to_string(), 0.72),
            ],
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "model.selected");
        match back {
            PipelineEvent::ModelSelected {
                model_name,
                score,
                all_scores,
            } => {
                assert_eq!(model_name, "gpt-4-turbo");
                assert!((score - 0.87_f32).abs() < 1e-5);
                assert_eq!(all_scores.len(), 2);
                assert_eq!(all_scores[0].0, "gpt-4-turbo");
                assert_eq!(all_scores[1].0, "claude-3-opus");
            }
            other => panic!("expected ModelSelected, got {:?}", other),
        }
    }

    #[test]
    fn test_model_selected_empty_all_scores_round_trip() {
        // Edge case: direct routing with a single model — all_scores may be empty.
        let event = PipelineEvent::ModelSelected {
            model_name: "direct-model".to_string(),
            score: 1.0,
            all_scores: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "model.selected");
        match back {
            PipelineEvent::ModelSelected {
                model_name,
                score,
                all_scores,
            } => {
                assert_eq!(model_name, "direct-model");
                assert!((score - 1.0_f32).abs() < 1e-5);
                assert!(all_scores.is_empty());
            }
            other => panic!("expected ModelSelected, got {:?}", other),
        }
    }

    #[test]
    fn test_commands_selected_round_trip() {
        let stubs = vec![
            weft_core::CommandStub {
                name: "web_search".to_string(),
                description: "Search the web".to_string(),
            },
            weft_core::CommandStub {
                name: "calculator".to_string(),
                description: "Evaluate math".to_string(),
            },
        ];
        let event = PipelineEvent::CommandsSelected {
            selected: stubs.clone(),
            candidates_scored: 5,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.selected");
        match back {
            PipelineEvent::CommandsSelected {
                selected,
                candidates_scored,
            } => {
                assert_eq!(selected.len(), 2);
                assert_eq!(selected[0].name, "web_search");
                assert_eq!(selected[1].name, "calculator");
                assert_eq!(candidates_scored, 5);
            }
            other => panic!("expected CommandsSelected, got {:?}", other),
        }
    }

    #[test]
    fn test_commands_selected_empty_round_trip() {
        // Edge case: no commands selected (below threshold).
        let event = PipelineEvent::CommandsSelected {
            selected: vec![],
            candidates_scored: 3,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.selected");
        match back {
            PipelineEvent::CommandsSelected {
                selected,
                candidates_scored,
            } => {
                assert!(selected.is_empty());
                assert_eq!(candidates_scored, 3);
            }
            other => panic!("expected CommandsSelected, got {:?}", other),
        }
    }

    #[test]
    fn test_provider_resolved_round_trip() {
        let event = PipelineEvent::ProviderResolved {
            model_name: "gpt-4-turbo".to_string(),
            model_id: "gpt-4-turbo-2024-04-09".to_string(),
            provider_name: "openai".to_string(),
            capabilities: vec!["chat_completions".to_string(), "tool_calling".to_string()],
            max_tokens: 128_000,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "provider.resolved");
        match back {
            PipelineEvent::ProviderResolved {
                model_name,
                model_id,
                provider_name,
                capabilities,
                max_tokens,
            } => {
                assert_eq!(model_name, "gpt-4-turbo");
                assert_eq!(model_id, "gpt-4-turbo-2024-04-09");
                assert_eq!(provider_name, "openai");
                assert_eq!(capabilities, vec!["chat_completions", "tool_calling"]);
                assert_eq!(max_tokens, 128_000);
            }
            other => panic!("expected ProviderResolved, got {:?}", other),
        }
    }

    #[test]
    fn test_provider_resolved_default_capabilities_round_trip() {
        // Edge case: default capabilities when none registered.
        let event = PipelineEvent::ProviderResolved {
            model_name: "unknown-model".to_string(),
            model_id: "unknown-model-v1".to_string(),
            provider_name: "unknown".to_string(),
            capabilities: vec!["chat_completions".to_string()],
            max_tokens: 4096,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "provider.resolved");
        match back {
            PipelineEvent::ProviderResolved {
                capabilities,
                max_tokens,
                ..
            } => {
                assert_eq!(capabilities, vec!["chat_completions"]);
                assert_eq!(max_tokens, 4096);
            }
            other => panic!("expected ProviderResolved, got {:?}", other),
        }
    }

    #[test]
    fn test_system_prompt_assembled_round_trip() {
        let event = PipelineEvent::SystemPromptAssembled {
            prompt_length: 512,
            layer_count: 2,
            message_count: 5,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "system_prompt.assembled");
        match back {
            PipelineEvent::SystemPromptAssembled {
                prompt_length,
                layer_count,
                message_count,
            } => {
                assert_eq!(prompt_length, 512);
                assert_eq!(layer_count, 2);
                assert_eq!(message_count, 5);
            }
            other => panic!("expected SystemPromptAssembled, got {:?}", other),
        }
    }

    #[test]
    fn test_system_prompt_assembled_empty_prompt_round_trip() {
        // Edge case: empty gateway prompt, no caller system prompt.
        let event = PipelineEvent::SystemPromptAssembled {
            prompt_length: 0,
            layer_count: 0,
            message_count: 1,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "system_prompt.assembled");
        match back {
            PipelineEvent::SystemPromptAssembled { prompt_length, .. } => {
                assert_eq!(prompt_length, 0);
            }
            other => panic!("expected SystemPromptAssembled, got {:?}", other),
        }
    }

    #[test]
    fn test_commands_formatted_structured_round_trip() {
        let event = PipelineEvent::CommandsFormatted {
            format: CommandFormat::Structured,
            command_count: 3,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.formatted");
        match back {
            PipelineEvent::CommandsFormatted {
                format,
                command_count,
            } => {
                assert_eq!(format, CommandFormat::Structured);
                assert_eq!(command_count, 3);
            }
            other => panic!("expected CommandsFormatted, got {:?}", other),
        }
    }

    #[test]
    fn test_commands_formatted_prompt_injected_round_trip() {
        let event = PipelineEvent::CommandsFormatted {
            format: CommandFormat::PromptInjected,
            command_count: 2,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.formatted");
        match back {
            PipelineEvent::CommandsFormatted { format, .. } => {
                assert_eq!(format, CommandFormat::PromptInjected);
            }
            other => panic!("expected CommandsFormatted, got {:?}", other),
        }
    }

    #[test]
    fn test_commands_formatted_no_commands_round_trip() {
        let event = PipelineEvent::CommandsFormatted {
            format: CommandFormat::NoCommands,
            command_count: 0,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "commands.formatted");
        match back {
            PipelineEvent::CommandsFormatted {
                format,
                command_count,
            } => {
                assert_eq!(format, CommandFormat::NoCommands);
                assert_eq!(command_count, 0);
            }
            other => panic!("expected CommandsFormatted, got {:?}", other),
        }
    }

    #[test]
    fn test_sampling_updated_full_round_trip() {
        let event = PipelineEvent::SamplingUpdated {
            max_tokens: 4096,
            temperature: Some(0.7),
            top_p: Some(0.9),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "sampling.updated");
        match back {
            PipelineEvent::SamplingUpdated {
                max_tokens,
                temperature,
                top_p,
            } => {
                assert_eq!(max_tokens, 4096);
                assert!((temperature.unwrap() - 0.7_f32).abs() < 1e-5);
                assert!((top_p.unwrap() - 0.9_f32).abs() < 1e-5);
            }
            other => panic!("expected SamplingUpdated, got {:?}", other),
        }
    }

    #[test]
    fn test_sampling_updated_no_temperature_top_p_round_trip() {
        // Edge case: request specified no temperature or top_p.
        let event = PipelineEvent::SamplingUpdated {
            max_tokens: 2048,
            temperature: None,
            top_p: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "sampling.updated");
        match back {
            PipelineEvent::SamplingUpdated {
                max_tokens,
                temperature,
                top_p,
            } => {
                assert_eq!(max_tokens, 2048);
                assert!(temperature.is_none());
                assert!(top_p.is_none());
            }
            other => panic!("expected SamplingUpdated, got {:?}", other),
        }
    }

    #[test]
    fn test_command_format_all_variants_serialize() {
        // Verify all CommandFormat variants serialize and deserialize without error.
        let formats = vec![
            CommandFormat::Structured,
            CommandFormat::PromptInjected,
            CommandFormat::NoCommands,
        ];
        for format in &formats {
            let json = serde_json::to_string(format).unwrap();
            let back: CommandFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, format);
        }
    }

    #[test]
    fn test_event_type_string_phase1_variants() {
        // Verify all Phase 1 variants produce the correct event_type strings.
        assert_eq!(
            PipelineEvent::ModelSelected {
                model_name: "m".to_string(),
                score: 0.9,
                all_scores: vec![],
            }
            .event_type_string(),
            "model.selected"
        );
        assert_eq!(
            PipelineEvent::CommandsSelected {
                selected: vec![],
                candidates_scored: 0,
            }
            .event_type_string(),
            "commands.selected"
        );
        assert_eq!(
            PipelineEvent::ProviderResolved {
                model_name: "m".to_string(),
                model_id: "m-v1".to_string(),
                provider_name: "p".to_string(),
                capabilities: vec![],
                max_tokens: 4096,
            }
            .event_type_string(),
            "provider.resolved"
        );
        assert_eq!(
            PipelineEvent::SystemPromptAssembled {
                prompt_length: 0,
                layer_count: 0,
                message_count: 0,
            }
            .event_type_string(),
            "system_prompt.assembled"
        );
        assert_eq!(
            PipelineEvent::CommandsFormatted {
                format: CommandFormat::NoCommands,
                command_count: 0,
            }
            .event_type_string(),
            "commands.formatted"
        );
        assert_eq!(
            PipelineEvent::SamplingUpdated {
                max_tokens: 4096,
                temperature: None,
                top_p: None,
            }
            .event_type_string(),
            "sampling.updated"
        );
    }

    #[test]
    fn test_message_injected_system_prompt_assembly_round_trip() {
        let msg = make_message();
        let event = PipelineEvent::MessageInjected {
            message: msg.clone(),
            source: crate::event::MessageInjectionSource::SystemPromptAssembly,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "message.injected");
        match back {
            PipelineEvent::MessageInjected {
                source: crate::event::MessageInjectionSource::SystemPromptAssembly,
                ..
            } => {}
            other => panic!(
                "expected MessageInjected with SystemPromptAssembly source, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_message_injected_command_format_injection_round_trip() {
        let msg = make_message();
        let event = PipelineEvent::MessageInjected {
            message: msg.clone(),
            source: crate::event::MessageInjectionSource::CommandFormatInjection,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.event_type_string(), "message.injected");
        match back {
            PipelineEvent::MessageInjected {
                source: crate::event::MessageInjectionSource::CommandFormatInjection,
                ..
            } => {}
            other => panic!(
                "expected MessageInjected with CommandFormatInjection source, got {:?}",
                other
            ),
        }
    }

    /// Verify that the existing event_type_string_all_variants test still covers every variant.
    /// This test builds the new Phase 1 variants to ensure they can be constructed
    /// and produce non-empty type strings.
    #[test]
    fn test_event_type_string_phase1_variants_nonempty() {
        let phase1_events: Vec<PipelineEvent> = vec![
            PipelineEvent::ModelSelected {
                model_name: "gpt-4".to_string(),
                score: 0.9,
                all_scores: vec![("gpt-4".to_string(), 0.9)],
            },
            PipelineEvent::CommandsSelected {
                selected: vec![weft_core::CommandStub {
                    name: "search".to_string(),
                    description: "Search".to_string(),
                }],
                candidates_scored: 3,
            },
            PipelineEvent::ProviderResolved {
                model_name: "gpt-4".to_string(),
                model_id: "gpt-4-turbo-2024".to_string(),
                provider_name: "openai".to_string(),
                capabilities: vec!["chat_completions".to_string(), "tool_calling".to_string()],
                max_tokens: 128_000,
            },
            PipelineEvent::SystemPromptAssembled {
                prompt_length: 200,
                layer_count: 2,
                message_count: 4,
            },
            PipelineEvent::CommandsFormatted {
                format: CommandFormat::Structured,
                command_count: 1,
            },
            PipelineEvent::SamplingUpdated {
                max_tokens: 4096,
                temperature: Some(0.5),
                top_p: None,
            },
        ];
        for event in &phase1_events {
            let type_str = event.event_type_string();
            assert!(
                !type_str.is_empty(),
                "event_type_string should not be empty for {:?}",
                event
            );
        }
    }
}
