//! Shared test infrastructure for weft_reactor integration tests.
//!
//! This module provides `TestActivity`, a composable builder that replaces
//! hand-written activity stubs. All shared helpers are here; inline stubs
//! in individual test files have been eliminated.
//!
//! Dead code warnings are suppressed because each integration test file is a
//! separate compilation unit and only uses a subset of the items defined here.
#![allow(dead_code, unused_imports)]

mod behaviors;
mod builders;
pub mod events;

use std::sync::{Arc, Mutex, atomic::AtomicU32};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::event::{FailureDetail, PipelineEvent};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor_trait::{Criticality, ServiceLocator};

// Re-export event assertion helpers.
pub use events::{EventAssertions, EventPredicate, TestEventLogAssertExt};

// Re-export everything from submodules so `use harness::*` in test files
// gets the full public API without any import changes.
pub use builders::{
    EmittingBuilder,
    FailBuilder,
    GenerateBuilder,
    PerCallBuilder,
    // Registry / config helpers
    build_new_preloop_registry,
    build_registry,
    // Pre-loop stubs
    command_formatting_stub,
    command_selection_stub,
    // execute_command helpers
    failing_execute_command,
    hanging_execute_command,
    model_selection_stub,
    new_preloop_pipeline_config,
    pipeline_with_validate,
    preloop_registry,
    provider_resolution_stub,
    reactor_config,
    sampling_adjustment_stub,
    simple_pipeline_config,
    simple_registry,
    system_prompt_assembly_stub,
    test_event_log,
    test_request,
    validate_stub,
};

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
        behaviors::execute_behavior(
            &self.behavior,
            &self.activity_name,
            execution_id,
            input,
            services,
            event_tx,
            cancel,
        )
        .await;
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

    // ── Pre-loop stub convenience methods (delegates to builders module) ──────

    /// Stub for validate: emits ValidationPassed + CommandsAvailable.
    pub fn validate_stub() -> Arc<dyn Activity> {
        builders::validate_stub()
    }

    /// Stub for model_selection: emits ModelSelected with stub-model.
    pub fn model_selection_stub() -> Arc<dyn Activity> {
        builders::model_selection_stub()
    }

    /// Stub for command_selection: emits CommandsSelected (empty).
    pub fn command_selection_stub() -> Arc<dyn Activity> {
        builders::command_selection_stub()
    }

    /// Stub for provider_resolution: emits ProviderResolved with stub values.
    pub fn provider_resolution_stub() -> Arc<dyn Activity> {
        builders::provider_resolution_stub()
    }

    /// Stub for system_prompt_assembly: emits MessageInjected + SystemPromptAssembled.
    pub fn system_prompt_assembly_stub() -> Arc<dyn Activity> {
        builders::system_prompt_assembly_stub()
    }

    /// Stub for command_formatting: emits CommandsFormatted (NoCommands).
    pub fn command_formatting_stub() -> Arc<dyn Activity> {
        builders::command_formatting_stub()
    }

    /// Stub for sampling_adjustment: emits SamplingUpdated (max_tokens 4096).
    pub fn sampling_adjustment_stub() -> Arc<dyn Activity> {
        builders::sampling_adjustment_stub()
    }
}
