//! Activity trait, ActivityInput, RoutingSnapshot, and ActivityError.
//!
//! Activities are event producers. They receive an `mpsc::Sender<PipelineEvent>`
//! and push events onto the channel as they work. They do not return values
//! to the Reactor. The Reactor spawns activities and processes their events
//! as they arrive.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::budget::Budget;
use crate::event::PipelineEvent;
use crate::event_log::{EventLog, EventLogError};
use crate::execution::ExecutionId;
use crate::services::Services;

/// A unit of side-effecting work within a pipeline.
///
/// Activities are event producers. They receive an `mpsc::Sender<PipelineEvent>`
/// and push events onto the channel as they work. They do NOT return values
/// to the Reactor. The Reactor spawns activities and processes their events
/// as they arrive.
///
/// Activities are self-recording: they push their own lifecycle events
/// (ActivityStarted, ActivityCompleted) and domain events (GenerationStarted,
/// CommandStarted, etc.) onto the channel.
///
/// Activities do NOT know about pipeline ordering or composition. They know
/// how to do one thing. The Reactor decides when to spawn them.
///
/// Object safety: the trait is object-safe (`Arc<dyn Activity>` works).
/// `execute` takes `mpsc::Sender<PipelineEvent>` and `CancellationToken`
/// by value (both are `Clone`), and borrows `ExecutionId`, `ActivityInput`,
/// `Services`, and `dyn EventLog`.
#[async_trait::async_trait]
pub trait Activity: Send + Sync {
    /// Human-readable name for this activity type.
    /// Used as the key in the ActivityRegistry and in event payloads.
    fn name(&self) -> &str;

    /// Execute the activity, pushing events onto the channel.
    ///
    /// `execution_id`: The execution this activity belongs to.
    /// `input`: Everything the activity needs to do its work.
    /// `services`: Shared infrastructure (providers, router, hooks, etc.).
    /// `event_log`: Where to read historical events for state reconstruction.
    /// `event_tx`: Channel sender. Push events here as they happen.
    /// `cancel`: Cancellation token for cooperative shutdown.
    ///
    /// No return type. The activity communicates exclusively through events.
    /// On completion, the activity pushes `ActivityCompleted`.
    /// On failure, the activity pushes `ActivityFailed`.
    async fn execute(
        &self,
        execution_id: &ExecutionId,
        input: ActivityInput,
        services: &Services,
        event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    );
}

/// Input to an activity. Contains the execution state the activity needs.
///
/// Not all fields are relevant to every activity. ValidateActivity needs
/// `messages` and `request`. GenerateActivity needs `messages`,
/// `routing_result`, and `budget`. Activities read what they need and
/// ignore the rest.
///
/// This is a value type, not a reference. Activities receive owned data
/// so they don't hold borrows across await points.
#[derive(Debug, Clone)]
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
    /// See Section 3.10 of the spec for idempotency semantics.
    pub idempotency_key: Option<String>,
    /// Accumulated token usage across all generation calls so far.
    pub accumulated_usage: weft_core::WeftUsage,
}

/// Snapshot of routing decisions. Populated by RouteActivity.
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
/// as the matcher target. Existing hook configs keyed on `matcher = "model"` or
/// `matcher = "commands"` therefore work without changes.
///
/// This trait extends [`Activity`]. It is a separate trait (rather than an
/// optional method on `Activity`) so that the compiler enforces the contract
/// and non-selection activities are not required to implement it.
///
/// # Object safety and downcasting
///
/// The trait is object-safe. However, downcasting `Arc<dyn Activity>` to check
/// whether an activity implements `SemanticSelection` requires `Any`. The
/// reactor avoids this by wiring selection activities explicitly — it knows at
/// wiring time which activities are selection activities. The trait exists for
/// documentation, type-level correctness, and future dynamic registration.
pub trait SemanticSelection: Activity {
    /// The routing domain this selection covers.
    ///
    /// Used as the hook matcher target for PreRoute/PostRoute hooks fired
    /// around this activity. Returns a `'static` str so no allocation
    /// is needed when passing it to `services.hooks.run_chain()`.
    ///
    /// Current values: `"model"` (ModelSelection), `"commands"` (CommandSelection).
    fn selection_domain(&self) -> &'static str;
}

/// Activity execution error.
///
/// Used internally by activities. Activities push `ActivityFailed`
/// events onto the channel rather than returning errors. This error
/// type exists for internal activity logic where Result patterns
/// are convenient, but it is never returned from `Activity::execute`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    // ── Object safety and basic Activity impl ───────────────────────────────

    /// Test that Activity is object-safe by creating `Arc<dyn Activity>`.
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
            _services: &Services,
            _event_log: &dyn EventLog,
            _event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            // no-op
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

    // ── Event-pushing activity ──────────────────────────────────────────────
    // Spec test expectation (line 2746): test that a simple test struct impl works
    // (pushes events onto channel).

    /// Activity that pushes one ActivityCompleted event onto the channel.
    struct EventPushingActivity;

    #[async_trait::async_trait]
    impl Activity for EventPushingActivity {
        fn name(&self) -> &str {
            "event_pusher"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &Services,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            let event = PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms: 1,
                idempotency_key: None,
            };
            // Send the event; ignore send errors (receiver may have dropped in tests).
            let _ = event_tx.send(event).await;
        }
    }

    /// No-op EventLog for tests that don't exercise the event log.
    struct NullEventLog;

    #[async_trait::async_trait]
    impl EventLog for NullEventLog {
        async fn create_execution(
            &self,
            _execution: &crate::execution::Execution,
        ) -> Result<(), EventLogError> {
            Ok(())
        }
        async fn update_execution_status(
            &self,
            _execution_id: &ExecutionId,
            _status: crate::execution::ExecutionStatus,
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

    #[tokio::test]
    async fn activity_pushes_event_onto_channel() {
        // Create a bounded channel and run the activity.
        let (tx, mut rx) = mpsc::channel::<PipelineEvent>(8);
        let cancel = CancellationToken::new();
        let exec_id = ExecutionId::new();

        // Build a minimal ActivityInput. Fields not used by EventPushingActivity
        // are set to their simplest valid values.
        let input = make_minimal_input();

        // Build a real Services with stub/null implementations. EventPushingActivity
        // never calls into any service, but the signature requires &Services.
        let services = make_stub_services();
        let log = NullEventLog;

        let activity = EventPushingActivity;
        activity
            .execute(&exec_id, input, &services, &log, tx, cancel)
            .await;

        // Verify that exactly one event arrived on the receiver.
        let event = rx.try_recv().expect("expected one event on channel");
        assert_eq!(event.event_type_string(), "activity.completed");
        match event {
            PipelineEvent::ActivityCompleted { name, .. } => {
                assert_eq!(name, "event_pusher");
            }
            other => panic!("expected ActivityCompleted, got {other:?}"),
        }

        // No additional events should be present.
        assert!(
            rx.try_recv().is_err(),
            "expected no more events after ActivityCompleted"
        );
    }

    // ── ActivityInput idempotency_key ───────────────────────────────────────
    // Spec test expectation (line 2747): ActivityInput: test that idempotency_key
    // field is present and optional.

    #[test]
    fn activity_input_idempotency_key_some() {
        let input = make_minimal_input_with_key(Some("exec:validate:0".to_string()));
        assert_eq!(input.idempotency_key, Some("exec:validate:0".to_string()));
    }

    #[test]
    fn activity_input_idempotency_key_none() {
        let input = make_minimal_input_with_key(None);
        assert!(input.idempotency_key.is_none());
    }

    // ── ActivityError display ───────────────────────────────────────────────

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
        use crate::event_log::EventLogError;
        let source = EventLogError::Storage("disk full".to_string());
        let err = ActivityError::EventLog {
            name: "generate".to_string(),
            source,
        };
        let msg = err.to_string();
        assert!(msg.contains("generate"));
        // The source is accessible via std::error::Error::source()
        use std::error::Error;
        assert!(err.source().is_some());
    }

    // ── RoutingSnapshot ─────────────────────────────────────────────────────

    #[test]
    fn routing_snapshot_serializes() {
        use weft_core::RoutingActivity;
        let snap = RoutingSnapshot {
            model_routing: RoutingActivity {
                model: "claude-3-haiku".to_string(),
                score: 0.95,
                filters: vec![],
            },
            tool_necessity: Some("high".to_string()),
            tool_necessity_score: Some(0.87),
        };
        let json = serde_json::to_string(&snap).expect("should serialize");
        let back: RoutingSnapshot = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(back.model_routing.model, "claude-3-haiku");
        assert_eq!(back.tool_necessity, Some("high".to_string()));
    }

    #[test]
    fn routing_snapshot_optional_fields_default_none() {
        use weft_core::RoutingActivity;
        let snap = RoutingSnapshot {
            model_routing: RoutingActivity {
                model: "gpt-4".to_string(),
                score: 0.0,
                filters: vec![],
            },
            tool_necessity: None,
            tool_necessity_score: None,
        };
        assert!(snap.tool_necessity.is_none());
        assert!(snap.tool_necessity_score.is_none());
    }

    // ── Test stubs ───────────────────────────────────────────────────────────

    /// Stub LLM provider that panics if actually called.
    /// Used in tests where no LLM calls are expected.
    struct PanicProvider;

    #[async_trait::async_trait]
    impl weft_llm::Provider for PanicProvider {
        async fn execute(
            &self,
            _request: weft_llm::ProviderRequest,
        ) -> Result<weft_llm::ProviderResponse, weft_llm::ProviderError> {
            panic!("PanicProvider: execute called unexpectedly in test")
        }

        fn name(&self) -> &str {
            "panic-provider"
        }
    }

    /// Stub semantic router that panics if actually called.
    struct PanicRouter;

    #[async_trait::async_trait]
    impl weft_router::SemanticRouter for PanicRouter {
        async fn route(
            &self,
            _user_message: &str,
            _domains: &[(
                weft_router::RoutingDomainKind,
                Vec<weft_router::RoutingCandidate>,
            )],
        ) -> Result<weft_router::RoutingDecision, weft_router::RouterError> {
            panic!("PanicRouter: route called unexpectedly in test")
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            _candidates: &[weft_router::RoutingCandidate],
        ) -> Result<Vec<weft_router::ScoredCandidate>, weft_router::RouterError> {
            panic!("PanicRouter: score_memory_candidates called unexpectedly in test")
        }
    }

    /// Stub command registry that panics if actually called.
    struct PanicCommandRegistry;

    #[async_trait::async_trait]
    impl weft_commands::CommandRegistry for PanicCommandRegistry {
        async fn list_commands(
            &self,
        ) -> Result<Vec<weft_core::CommandStub>, weft_commands::CommandError> {
            panic!("PanicCommandRegistry: list_commands called unexpectedly in test")
        }

        async fn describe_command(
            &self,
            _name: &str,
        ) -> Result<weft_core::CommandDescription, weft_commands::CommandError> {
            panic!("PanicCommandRegistry: describe_command called unexpectedly in test")
        }

        async fn execute_command(
            &self,
            _invocation: &weft_core::CommandInvocation,
        ) -> Result<weft_core::CommandResult, weft_commands::CommandError> {
            panic!("PanicCommandRegistry: execute_command called unexpectedly in test")
        }
    }

    /// Build a minimal Services for tests that do not call any service methods.
    fn make_stub_services() -> Services {
        // Parse a minimal WeftConfig from TOML — the cheapest way to get a valid config.
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

        // Build a ProviderRegistry with a single stub provider.
        let mut providers: HashMap<String, Arc<dyn weft_llm::Provider>> = HashMap::new();
        providers.insert("stub".to_string(), Arc::new(PanicProvider));
        let mut model_ids = HashMap::new();
        model_ids.insert("stub".to_string(), "stub-model".to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert("stub".to_string(), 4096u32);
        let mut capabilities: HashMap<String, HashSet<weft_llm::Capability>> = HashMap::new();
        capabilities.insert("stub".to_string(), HashSet::new());
        let registry = weft_llm::ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            capabilities,
            "stub".to_string(),
        );

        Services {
            config: Arc::new(config),
            providers: Arc::new(registry),
            router: Arc::new(PanicRouter),
            commands: Arc::new(PanicCommandRegistry),
            memory: None,
            hooks: Arc::new(weft_hooks::NullHookRunner),
            reactor_handle: std::sync::OnceLock::new(),
            request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn make_minimal_input() -> ActivityInput {
        make_minimal_input_with_key(None)
    }

    fn make_minimal_input_with_key(idempotency_key: Option<String>) -> ActivityInput {
        use chrono::Utc;
        use weft_core::{
            ModelRoutingInstruction, Role, SamplingOptions, Source, WeftMessage, WeftRequest,
        };

        ActivityInput {
            messages: vec![WeftMessage {
                role: Role::User,
                source: Source::Client,
                model: None,
                content: vec![weft_core::ContentPart::Text("hello".to_string())],
                delta: false,
                message_index: 0,
            }],
            request: WeftRequest {
                messages: vec![],
                routing: ModelRoutingInstruction::parse("auto"),
                options: SamplingOptions::default(),
            },
            routing_result: None,
            budget: crate::budget::Budget::new(10, 5, 3, Utc::now() + chrono::Duration::hours(1)),
            metadata: serde_json::Value::Null,
            generation_config: None,
            accumulated_text: String::new(),
            available_commands: vec![],
            idempotency_key,
            accumulated_usage: weft_core::WeftUsage::default(),
        }
    }
}
