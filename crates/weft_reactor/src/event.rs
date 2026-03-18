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
}
