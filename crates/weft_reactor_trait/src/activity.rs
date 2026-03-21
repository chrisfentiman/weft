//! Activity trait, ActivityInput, RoutingSnapshot, SemanticSelection, and ActivityError.
//!
//! Activities are event producers. They receive an `mpsc::Sender<PipelineEvent>`
//! and push events onto the channel as they work. They do not return values
//! to the Reactor. The Reactor spawns activities and processes their events
//! as they arrive.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::budget::Budget;
use crate::event::PipelineEvent;
use crate::event_log::{EventLog, EventLogError};
use crate::execution::ExecutionId;
use crate::service::{ChildSpawner, ServiceLocator};

/// A unit of side-effecting work within a pipeline.
///
/// Activities are event producers. They receive an `mpsc::Sender<PipelineEvent>`
/// and push events onto the channel as they work. They do NOT return values
/// to the Reactor. The Reactor spawns activities and processes their events
/// as they arrive.
///
/// Activities are self-recording: they push their own lifecycle events
/// (`Activity(ActivityEvent::Started)`, `Activity(ActivityEvent::Completed)`)
/// and domain events onto the channel.
///
/// Activities do NOT know about pipeline ordering or composition. They know
/// how to do one thing. The Reactor decides when to spawn them.
///
/// Object safety: the trait is object-safe (`Arc<dyn Activity>` works).
/// `execute` takes `mpsc::Sender<PipelineEvent>` and `CancellationToken`
/// by value (both are `Clone`), and borrows `ExecutionId`, `ActivityInput`,
/// `&dyn ServiceLocator`, and `&dyn EventLog`.
#[async_trait::async_trait]
pub trait Activity: Send + Sync {
    /// Human-readable name for this activity type.
    /// Used as the key in the ActivityRegistry and in event payloads.
    fn name(&self) -> &str;

    /// Execute the activity, pushing events onto the channel.
    ///
    /// `execution_id`: The execution this activity belongs to.
    /// `input`: Everything the activity needs to do its work.
    /// `services`: Shared infrastructure via `ServiceLocator` trait object.
    /// `event_log`: Where to read historical events for state reconstruction.
    /// `event_tx`: Channel sender. Push events here as they happen.
    /// `cancel`: Cancellation token for cooperative shutdown.
    ///
    /// No return type. The activity communicates exclusively through events.
    /// On completion, the activity pushes `Activity(ActivityEvent::Completed)`.
    /// On failure, the activity pushes `Activity(ActivityEvent::Failed)`.
    async fn execute(
        &self,
        execution_id: &ExecutionId,
        input: ActivityInput,
        services: &dyn ServiceLocator,
        event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    );
}

/// Input to an activity. Contains the execution state the activity needs.
///
/// Not all fields are relevant to every activity. `ValidateActivity` needs
/// `messages` and `request`. `GenerateActivity` needs `messages`,
/// `routing_result`, and `budget`. Activities read what they need and
/// ignore the rest.
///
/// This is a value type, not a reference. Activities receive owned data
/// so they don't hold borrows across await points.
///
/// `Debug` is implemented manually because `dyn ChildSpawner` is not `Debug`.
#[derive(Clone)]
pub struct ActivityInput {
    /// The conversation messages as of this point in the execution.
    pub messages: Vec<weft_core::WeftMessage>,
    /// The original request (for activities that need routing info, options, etc.).
    pub request: weft_core::WeftRequest,
    /// The current routing result, if routing has completed.
    pub routing_result: Option<RoutingSnapshot>,
    /// The current budget state.
    pub budget: Budget,
    /// Activity-specific metadata. Used for passing configuration
    /// from the pipeline config to the activity.
    pub metadata: serde_json::Value,
    /// The generation config to use (set by routing, potentially
    /// overridden by ForceGenerationConfig signal).
    pub generation_config: Option<serde_json::Value>,
    /// Accumulated text content from GeneratedEvent::Content events.
    pub accumulated_text: String,
    /// Available commands, populated by ValidateActivity.
    pub available_commands: Vec<weft_core::CommandStub>,
    /// Idempotency key for deduplicating activity executions.
    /// Format: `{execution_id}:{activity_name}:{iteration}`.
    /// The Reactor checks the EventLog for an existing completion
    /// event with this key before executing the activity.
    pub idempotency_key: Option<String>,
    /// Accumulated token usage across all generation calls so far.
    pub accumulated_usage: weft_core::WeftUsage,
    /// Optional child execution spawner.
    ///
    /// Populated for `GenerateActivity`. `None` for all other activities.
    /// `GenerateActivity` uses this instead of accessing `services.reactor_handle`
    /// directly, since the Activity trait no longer receives a concrete `Services`.
    pub child_spawner: Option<Arc<dyn ChildSpawner>>,
}

impl std::fmt::Debug for ActivityInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivityInput")
            .field("messages", &self.messages)
            .field("request", &self.request)
            .field("routing_result", &self.routing_result)
            .field("budget", &self.budget)
            .field("metadata", &self.metadata)
            .field("generation_config", &self.generation_config)
            .field("accumulated_text", &self.accumulated_text)
            .field("available_commands", &self.available_commands)
            .field("idempotency_key", &self.idempotency_key)
            .field("accumulated_usage", &self.accumulated_usage)
            .field(
                "child_spawner",
                &self.child_spawner.as_ref().map(|_| "<ChildSpawner>"),
            )
            .finish()
    }
}

/// Snapshot of routing decisions. Populated by ModelSelectionActivity.
///
/// Uses `RoutingActivity` from Weft Wire for the model routing result.
/// Additional routing domains (tool_necessity, memory) are carried as
/// separate fields since they don't map to the RoutingActivity type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutingSnapshot {
    /// The model routing result as a Weft Wire RoutingActivity.
    pub model_routing: weft_core::RoutingActivity,
    pub tool_necessity: Option<String>,
    pub tool_necessity_score: Option<f32>,
}

/// An Activity that performs semantic selection: scoring input against candidates.
///
/// Activities 1–2 (ModelSelection, CommandSelection) implement this trait.
/// The reactor fires PreRoute/PostRoute hooks around any activity that
/// implements `SemanticSelection`, using [`SemanticSelection::selection_domain`]
/// as the matcher target.
///
/// This trait extends [`Activity`]. It is a separate trait so that the compiler
/// enforces the contract and non-selection activities are not required to implement it.
pub trait SemanticSelection: Activity {
    /// The routing domain this selection covers.
    ///
    /// Used as the hook matcher target for PreRoute/PostRoute hooks fired
    /// around this activity. Returns a `'static` str so no allocation
    /// is needed when passing it to `services.hooks().run_chain()`.
    ///
    /// Current values: `"model"` (ModelSelection), `"commands"` (CommandSelection).
    fn selection_domain(&self) -> &'static str;
}

/// Activity execution error.
///
/// Used internally by activities. Activities push `Activity(ActivityEvent::Failed)`
/// events onto the channel rather than returning errors. This error type exists
/// for internal activity logic where Result patterns are convenient.
#[derive(Debug, thiserror::Error)]
pub enum ActivityError {
    #[error("activity '{name}' failed: {reason}")]
    Failed { name: String, reason: String },

    #[error("activity '{name}' cancelled")]
    Cancelled { name: String },

    #[error("activity '{name}' timed out")]
    Timeout { name: String },

    #[error("activity '{name}' received invalid input: {reason}")]
    InvalidInput { name: String, reason: String },

    #[error("event log error in activity '{name}': {source}")]
    EventLog {
        name: String,
        #[source]
        source: EventLogError,
    },
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::sync::Arc;

    // ── Object safety ─────────────────────────────────────────────────────

    struct NoOpActivity;

    #[async_trait::async_trait]
    impl Activity for NoOpActivity {
        fn name(&self) -> &str {
            "noop"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn ServiceLocator,
            _event_log: &dyn EventLog,
            _event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
        }
    }

    #[test]
    fn activity_is_object_safe() {
        let _arc: Arc<dyn Activity> = Arc::new(NoOpActivity);
    }

    #[test]
    fn activity_name_returns_correct_value() {
        let a = NoOpActivity;
        assert_eq!(a.name(), "noop");
    }

    // ── ActivityInput child_spawner field ─────────────────────────────────

    #[test]
    fn activity_input_child_spawner_none_by_default() {
        let input = make_minimal_input(None, None);
        assert!(input.child_spawner.is_none());
    }

    #[test]
    fn activity_input_child_spawner_some() {
        struct NopSpawner;

        #[async_trait::async_trait]
        impl ChildSpawner for NopSpawner {
            async fn spawn_child(
                &self,
                req: crate::service::SpawnRequest,
                _: Option<&tokio_util::sync::CancellationToken>,
            ) -> Result<Budget, String> {
                Ok(req.parent_budget)
            }
        }

        let spawner: Arc<dyn ChildSpawner> = Arc::new(NopSpawner);
        let input = make_minimal_input(None, Some(spawner));
        assert!(input.child_spawner.is_some());
    }

    // ── ActivityInput idempotency_key ─────────────────────────────────────

    #[test]
    fn activity_input_idempotency_key_some() {
        let input = make_minimal_input(Some("exec:validate:0".to_string()), None);
        assert_eq!(input.idempotency_key, Some("exec:validate:0".to_string()));
    }

    #[test]
    fn activity_input_idempotency_key_none() {
        let input = make_minimal_input(None, None);
        assert!(input.idempotency_key.is_none());
    }

    // ── ActivityError display ─────────────────────────────────────────────

    #[test]
    fn activity_error_display_failed() {
        let err = ActivityError::Failed {
            name: "generate".to_string(),
            reason: "provider unavailable".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("generate"));
        assert!(msg.contains("provider unavailable"));
    }

    #[test]
    fn activity_error_display_cancelled() {
        let err = ActivityError::Cancelled {
            name: "validate".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("validate"));
        assert!(msg.contains("cancelled"));
    }

    #[test]
    fn activity_error_display_timeout() {
        let err = ActivityError::Timeout {
            name: "execute_command".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("execute_command"));
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn activity_error_display_invalid_input() {
        let err = ActivityError::InvalidInput {
            name: "route".to_string(),
            reason: "missing messages".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("route"));
        assert!(msg.contains("missing messages"));
    }

    #[test]
    fn activity_error_event_log_source() {
        use std::error::Error;

        let source = EventLogError::Storage("disk full".to_string());
        let err = ActivityError::EventLog {
            name: "generate".to_string(),
            source,
        };
        let msg = err.to_string();
        assert!(msg.contains("generate"));
        assert!(err.source().is_some());
    }

    // ── RoutingSnapshot serde ─────────────────────────────────────────────

    #[test]
    fn routing_snapshot_serializes() {
        let snap = RoutingSnapshot {
            model_routing: weft_core::RoutingActivity {
                model: "claude-3-haiku".to_string(),
                score: 0.95,
                filters: vec![],
            },
            tool_necessity: Some("high".to_string()),
            tool_necessity_score: Some(0.87),
        };
        let json = serde_json::to_string(&snap).expect("must serialize");
        let back: RoutingSnapshot = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(back.model_routing.model, "claude-3-haiku");
        assert_eq!(back.tool_necessity, Some("high".to_string()));
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_minimal_input(
        idempotency_key: Option<String>,
        child_spawner: Option<Arc<dyn ChildSpawner>>,
    ) -> ActivityInput {
        use chrono::Utc;
        use weft_core::{
            ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftMessage,
            WeftRequest,
        };

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
            routing_result: None,
            budget: Budget::new(10, 5, 3, Utc::now() + chrono::Duration::hours(1)),
            metadata: serde_json::Value::Null,
            generation_config: None,
            accumulated_text: String::new(),
            available_commands: vec![],
            idempotency_key,
            accumulated_usage: weft_core::WeftUsage::default(),
            child_spawner,
        }
    }
}
