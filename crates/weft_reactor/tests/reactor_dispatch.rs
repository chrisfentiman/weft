//! Reactor dispatch integration tests.
//!
//! These tests exercise the full dispatch loop using stub services and
//! `weft_eventlog_memory::InMemoryEventLog`. Time-sensitive tests use
//! `tokio::time::pause()` and `tokio::time::advance()`.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::budget::Budget;
use weft_reactor::config::{
    ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig, RetryPolicy,
};
use weft_reactor::error::ReactorError;
use weft_reactor::event::{
    ActivityEvent, CommandEvent, ContextEvent, ExecutionEvent, GeneratedEvent, GenerationEvent,
    HookOutcome, PipelineEvent, SignalEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::{Execution, ExecutionId, ExecutionStatus};
use weft_reactor::reactor::Reactor;
use weft_reactor::registry::ActivityRegistry;
use weft_reactor::signal::Signal;
use weft_reactor::test_support::{
    TestEventLog, make_test_services, make_test_services_with_blocking_hook,
    make_test_services_with_response,
};
use weft_reactor::{RequestId, TenantId};

use weft_core::{
    CommandAction, CommandInvocation, ContentPart, HookEvent, ModelRoutingInstruction, Role,
    SamplingOptions, Source, WeftMessage, WeftRequest,
};

// ── Test activity stubs ───────────────────────────────────────────────────

/// Activity that immediately pushes Done (no commands, no content).
struct ImmediateDoneActivity {
    name: String,
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
struct TextGenerateActivity {
    name: String,
    response_text: String,
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
struct AlwaysFailActivity {
    name: String,
    retryable: bool,
    error_msg: String,
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
            }))
            .await;
    }
}

/// Activity that fails N times then succeeds on the (N+1)th call.
struct FailThenSucceedActivity {
    name: String,
    fails_remaining: std::sync::atomic::AtomicU32,
}

impl FailThenSucceedActivity {
    fn new(name: &str, fail_count: u32) -> Self {
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
struct NoOpActivity {
    name: String,
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
struct StubAssembleResponse {
    name: String,
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
struct StubExecuteCommand {
    name: String,
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

// ── Reactor builder helpers ───────────────────────────────────────────────

/// Build a simple pipeline config with a single generate activity and no pre/post activities.
fn simple_pipeline_config(generate_name: &str) -> PipelineConfig {
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
fn pipeline_with_validate(generate_name: &str) -> PipelineConfig {
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
fn reactor_config(pipeline: PipelineConfig) -> ReactorConfig {
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
fn test_request() -> WeftRequest {
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
fn build_registry(activities: Vec<Arc<dyn Activity>>) -> Arc<ActivityRegistry> {
    let mut registry = ActivityRegistry::new();
    for activity in activities {
        registry
            .register(activity)
            .expect("duplicate activity name in test");
    }
    Arc::new(registry)
}

/// Build a TestEventLog for testing.
fn test_event_log() -> Arc<TestEventLog> {
    Arc::new(TestEventLog::new())
}

// ── Construction tests ────────────────────────────────────────────────────

#[test]
fn reactor_new_missing_default_pipeline_returns_error() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "other".to_string(), // not "default"
            pre_loop: vec![],
            post_loop: vec![],
            generate: ActivityRef::Name("generate".to_string()),
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let result = Reactor::new(services, event_log, registry, &config);
    assert!(
        matches!(result, Err(ReactorError::Config(_))),
        "expected Config error for missing default pipeline"
    );
}

#[test]
fn reactor_new_unknown_activity_returns_error() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    // Registry has no "nonexistent" activity.
    let registry = build_registry(vec![
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("nonexistent".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    });

    let result = Reactor::new(services, event_log, registry, &config);
    assert!(
        matches!(result, Err(ReactorError::ActivityNotFound(_))),
        "expected ActivityNotFound error"
    );
}

#[test]
fn reactor_new_activity_with_retry_resolves_correctly() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 3,
                    initial_backoff_ms: 100,
                    max_backoff_ms: 1000,
                    backoff_multiplier: 2.0,
                }),
                timeout_secs: Some(60),
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let result = Reactor::new(services, event_log, registry, &config);
    assert!(
        result.is_ok(),
        "should construct with retry policy: {:?}",
        result
    );
}

// ── Simple request/response test ─────────────────────────────────────────

#[tokio::test]
async fn simple_request_response_completes() {
    let services = Arc::new(make_test_services_with_response("Hello, world!"));
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(TextGenerateActivity {
            name: "generate".to_string(),
            response_text: "Hello, world!".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _signal_tx) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    // Verify response contains the generated text.
    let has_text = result.response.messages.iter().any(|m| {
        m.content
            .iter()
            .any(|p| matches!(p, ContentPart::Text(t) if t.contains("Hello, world!")))
    });
    assert!(has_text, "response should contain generated text");

    // Verify budget_used is sane.
    assert_eq!(result.budget_used.generation_calls, 1);

    // Verify event log contains execution.started and execution.completed.
    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();
    let has_started = events.iter().any(|e| e.event_type == "execution.started");
    let has_completed = events.iter().any(|e| e.event_type == "execution.completed");
    assert!(has_started, "event log should contain execution.started");
    assert!(
        has_completed,
        "event log should contain execution.completed"
    );
}

#[tokio::test]
async fn pre_loop_activity_runs_before_generate() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(NoOpActivity {
            name: "validate".to_string(),
        }),
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(pipeline_with_validate("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();
    // validate should come before generation.started in the log.
    // PipelineEvent serializes with adjacent tagging: {"category": "Activity", "event": {"type": "Started", "name": "..."}}
    let validate_pos = events.iter().position(|e| {
        e.event_type == "activity.started"
            && e.payload
                .get("event")
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
                == Some("validate")
    });
    let gen_pos = events
        .iter()
        .position(|e| e.event_type == "generation.started");

    assert!(validate_pos.is_some(), "validate activity should be in log");
    assert!(gen_pos.is_some(), "generation.started should be in log");
    assert!(
        validate_pos.unwrap() < gen_pos.unwrap(),
        "validate should run before generation"
    );
}

// ── Budget exhaustion ─────────────────────────────────────────────────────

#[tokio::test]
async fn budget_exhaustion_terminates_gracefully() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    // max_generation_calls = 1 means after 1 call, budget is exhausted.
    let config = ReactorConfig {
        pipelines: vec![simple_pipeline_config("generate")],
        budget: BudgetConfig {
            max_generation_calls: 1,
            max_iterations: 10,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    // Execution should succeed (budget exhaustion is not an error, it's graceful termination).
    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        result.is_ok(),
        "budget exhaustion should return Ok with partial results, got: {:?}",
        result
    );
    let (execution_result, _) = result.unwrap();
    assert_eq!(execution_result.budget_used.generation_calls, 1);
}

// ── Cancellation ──────────────────────────────────────────────────────────

/// Cancel signal pushed onto the event channel terminates execution.
#[tokio::test]
async fn cancel_signal_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    struct CancelViaChannelActivity;

    #[async_trait::async_trait]
    impl Activity for CancelViaChannelActivity {
        fn name(&self) -> &str {
            "generate"
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
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Signal(SignalEvent::Received(
                    Signal::Cancel {
                        reason: "test cancel via channel".to_string(),
                    },
                )))
                .await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(CancelViaChannelActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _signal_tx) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("cancelled execution should return Ok");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        event_types.contains(&"execution.cancelled"),
        "event log should contain execution.cancelled; got: {:?}",
        event_types
    );
}

/// Cancellation via Signal::Cancel sent on the event channel.
#[tokio::test]
async fn cancel_signal_on_channel_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    struct CancellingGenerateActivity;

    #[async_trait::async_trait]
    impl Activity for CancellingGenerateActivity {
        fn name(&self) -> &str {
            "generate"
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
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Signal(SignalEvent::Received(
                    Signal::Cancel {
                        reason: "test cancel".to_string(),
                    },
                )))
                .await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(CancellingGenerateActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        result.is_ok(),
        "cancel should return Ok with partial results: {:?}",
        result
    );

    let (exec_result, _) = result.unwrap();
    let events = event_log
        .read(&exec_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let cancelled = events.iter().any(|e| e.event_type == "execution.cancelled");
    assert!(cancelled, "event log should contain execution.cancelled");
}

// ── Hook blocking ─────────────────────────────────────────────────────────

#[tokio::test]
async fn hook_block_in_pre_loop_returns_hook_blocked_error() {
    let services = Arc::new(make_test_services_with_blocking_hook(
        HookEvent::RequestStart,
        "blocked by policy",
    ));
    let event_log = test_event_log();

    struct HookStartActivity;

    #[async_trait::async_trait]
    impl Activity for HookStartActivity {
        fn name(&self) -> &str {
            "hook_request_start"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name().to_string(),
                }))
                .await;

            let result = services
                .hooks()
                .run_chain(HookEvent::RequestStart, serde_json::json!({}), None)
                .await;

            match result {
                weft_hooks::HookChainResult::Blocked { hook_name, reason } => {
                    let _ = event_tx
                        .send(PipelineEvent::Hook(HookOutcome::Blocked {
                            hook_event: "request_start".to_string(),
                            hook_name,
                            reason,
                        }))
                        .await;
                }
                weft_hooks::HookChainResult::Allowed { .. } => {
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Completed {
                            name: self.name().to_string(),
                            idempotency_key: None,
                        }))
                        .await;
                }
            }
        }
    }

    let registry = build_registry(vec![
        Arc::new(HookStartActivity),
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![ActivityRef::Name("hook_request_start".to_string())],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("generate".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    });

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        matches!(result, Err(ReactorError::HookBlocked { .. })),
        "expected HookBlocked error, got: {:?}",
        result
    );
}

// ── Activity retry ────────────────────────────────────────────────────────

#[tokio::test]
async fn generate_fails_once_then_succeeds_with_retry() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        Arc::new(FailThenSucceedActivity::new("generate", 1)),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 2,
                    initial_backoff_ms: 1, // 1ms for tests
                    max_backoff_ms: 10,
                    backoff_multiplier: 1.0,
                }),
                timeout_secs: None,
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        result.is_ok(),
        "execution should succeed after retry: {:?}",
        result
    );

    let exec_id = result.unwrap().0.execution_id;
    let events = event_log.read(&exec_id, None::<u64>).await.unwrap();
    let retried = events.iter().any(|e| e.event_type == "activity.retried");
    assert!(retried, "event log should contain activity.retried event");
}

#[tokio::test]
async fn generate_not_retried_when_retryable_false() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        Arc::new(AlwaysFailActivity {
            name: "generate".to_string(),
            retryable: false,
            error_msg: "auth error".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 3,
                    initial_backoff_ms: 1,
                    max_backoff_ms: 10,
                    backoff_multiplier: 1.0,
                }),
                timeout_secs: None,
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(result.is_err(), "non-retryable failure should return Err");
}

#[tokio::test]
async fn retry_exhaustion_returns_error() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        Arc::new(AlwaysFailActivity {
            name: "generate".to_string(),
            retryable: true,
            error_msg: "transient error".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 2,
                    initial_backoff_ms: 1,
                    max_backoff_ms: 5,
                    backoff_multiplier: 1.0,
                }),
                timeout_secs: None,
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 10,
            max_iterations: 10,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(result.is_err(), "exhausted retries should return Err");
}

#[tokio::test]
async fn retry_skipped_when_budget_exhausted() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        Arc::new(AlwaysFailActivity {
            name: "generate".to_string(),
            retryable: true,
            error_msg: "transient error".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 5,
                    initial_backoff_ms: 1,
                    max_backoff_ms: 5,
                    backoff_multiplier: 1.0,
                }),
                timeout_secs: None,
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 1,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    // Pass a pre-exhausted budget directly.
    let exhausted_budget = Budget::new(
        1,
        5,
        3,
        chrono::Utc::now() - chrono::Duration::seconds(1), // already past
    );

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            Some(exhausted_budget),
            None,
            None,
        )
        .await;

    let (exec_result, _) = result.expect("budget exhaustion should return Ok gracefully");

    let events = event_log
        .read(&exec_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();

    assert!(
        event_types.contains(&"budget.exhausted"),
        "event log should contain budget.exhausted when deadline is past; got: {event_types:?}"
    );

    assert!(
        !event_types.contains(&"activity.retried"),
        "no activity.retried should appear when budget is exhausted before generation; got: {event_types:?}"
    );
}

// ── Generation timeout ────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn generation_timeout_fires_after_silence() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    struct StallActivity;

    #[async_trait::async_trait]
    impl Activity for StallActivity {
        fn name(&self) -> &str {
            "generate"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "generate".to_string(),
                }))
                .await;
            cancel.cancelled().await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "generate".to_string(),
                    error: "cancelled".to_string(),
                    retryable: false,
                }))
                .await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(StallActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: None,
                timeout_secs: Some(5),
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 5,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let reactor_arc = Arc::new(reactor);
    let reactor_ref = Arc::clone(&reactor_arc);

    let handle = tokio::spawn(async move {
        reactor_ref
            .execute(
                test_request(),
                TenantId("tenant1".to_string()),
                RequestId("req1".to_string()),
                None,
                None,
                None,
                None,
            )
            .await
    });

    tokio::time::advance(Duration::from_secs(6)).await;

    let result = handle.await.expect("task should complete");

    assert!(
        result.is_err(),
        "generation timeout should fail execution without retry"
    );
}

// ── Heartbeat monitoring ──────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn heartbeat_miss_cancels_activity() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    struct HeartbeatThenStallActivity;

    #[async_trait::async_trait]
    impl Activity for HeartbeatThenStallActivity {
        fn name(&self) -> &str {
            "generate"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "generate".to_string(),
                }))
                .await;

            for _ in 0..2 {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Heartbeat {
                        activity_name: "generate".to_string(),
                    }))
                    .await;
            }

            cancel.cancelled().await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "generate".to_string(),
                    error: "cancelled".to_string(),
                    retryable: false,
                }))
                .await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(HeartbeatThenStallActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: None,
                timeout_secs: Some(300),
                heartbeat_interval_secs: Some(2),
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 300,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let reactor_arc = Arc::new(reactor);
    let reactor_ref = Arc::clone(&reactor_arc);

    let handle = tokio::spawn(async move {
        reactor_ref
            .execute(
                test_request(),
                TenantId("tenant1".to_string()),
                RequestId("req1".to_string()),
                None,
                None,
                None,
                None,
            )
            .await
    });

    tokio::time::advance(Duration::from_secs(10)).await;

    let result = handle.await.expect("task should complete");
    assert!(
        result.is_err(),
        "heartbeat miss should fail execution without retry: {:?}",
        result
    );
}

// ── Idempotency ───────────────────────────────────────────────────────────

#[tokio::test]
async fn idempotency_check_skips_already_completed_activity() {
    let event_log = test_event_log();
    let services = Arc::new(make_test_services());
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let execution_id = ExecutionId::new();
    let key = format!("{}:generate:0", execution_id);

    event_log
        .create_execution(&Execution {
            id: execution_id.clone(),
            tenant_id: TenantId("tenant1".to_string()),
            request_id: RequestId("req1".to_string()),
            parent_id: None,
            pipeline_name: "default".to_string(),
            status: ExecutionStatus::Running,
            created_at: chrono::Utc::now(),
            depth: 0,
        })
        .await
        .unwrap();

    event_log
        .append(
            &execution_id,
            "activity.started",
            serde_json::json!({ "name": "generate" }),
            1,
            None,
        )
        .await
        .unwrap();

    let done_ev = weft_reactor::event::PipelineEvent::Generation(GenerationEvent::Chunk(
        GeneratedEvent::Done,
    ));
    event_log
        .append(
            &execution_id,
            done_ev.event_type_string(),
            serde_json::to_value(&done_ev).unwrap(),
            1,
            None,
        )
        .await
        .unwrap();

    event_log
        .append(
            &execution_id,
            "activity.completed",
            serde_json::json!({
                "name": "generate",
                "duration_ms": 1,
                "idempotency_key": key
            }),
            1,
            Some(&key),
        )
        .await
        .unwrap();

    let result =
        weft_reactor::reactor::test_hooks::check_idempotency_pub(&reactor, &execution_id, &key)
            .await
            .expect("check_idempotency should not fail");

    assert!(
        result.is_some(),
        "idempotency check should return Some when key exists in event log"
    );

    let replayed = result.unwrap();
    assert!(
        !replayed.is_empty(),
        "replayed events slice should not be empty"
    );

    let types: Vec<&str> = replayed.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        types.contains(&"activity.started"),
        "replayed events should include activity.started; got: {types:?}"
    );
    assert!(
        types.contains(&"activity.completed"),
        "replayed events should include activity.completed; got: {types:?}"
    );

    let miss_key = format!("{}:generate:99", execution_id);
    let miss_result = weft_reactor::reactor::test_hooks::check_idempotency_pub(
        &reactor,
        &execution_id,
        &miss_key,
    )
    .await
    .expect("check_idempotency should not fail on miss");
    assert!(
        miss_result.is_none(),
        "idempotency check should return None when key is absent"
    );
}

// ── Streaming ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn generated_content_events_forwarded_to_client_tx() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        Arc::new(TextGenerateActivity {
            name: "generate".to_string(),
            response_text: "streaming token".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (client_tx, mut client_rx) = mpsc::channel::<PipelineEvent>(64);

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            Some(client_tx),
            None,
        )
        .await;

    assert!(
        result.is_ok(),
        "streaming execution should succeed: {:?}",
        result
    );

    let mut content_events = Vec::new();
    while let Ok(event) = client_rx.try_recv() {
        if matches!(
            &event,
            PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content { .. }))
        ) {
            content_events.push(event);
        }
    }

    assert!(
        !content_events.is_empty(),
        "client channel should receive Generated(Content) events"
    );
}

// ── Event log completeness ────────────────────────────────────────────────

#[tokio::test]
async fn event_log_contains_complete_execution_trace() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        Arc::new(TextGenerateActivity {
            name: "generate".to_string(),
            response_text: "test response".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();

    assert!(
        event_types.contains(&"execution.started"),
        "missing execution.started"
    );
    assert!(
        event_types.contains(&"generation.started"),
        "missing generation.started"
    );
    assert!(
        event_types.contains(&"generation.chunk"),
        "missing generation.chunk (content)"
    );
    assert!(
        event_types.contains(&"execution.completed"),
        "missing execution.completed"
    );

    let seqs: Vec<u64> = events.iter().map(|e| e.sequence).collect();
    for w in seqs.windows(2) {
        assert!(w[0] < w[1], "event sequences should be strictly increasing");
    }
}

// ── Phase 5: Child executions ─────────────────────────────────────────────

#[test]
fn oncelock_none_before_set_some_after() {
    let services = weft_reactor::test_support::make_test_services();
    assert!(
        services.reactor_handle.get().is_none(),
        "reactor_handle should be None before OnceLock::set"
    );

    let event_log: Arc<dyn weft_reactor::event_log::EventLog> =
        Arc::new(weft_reactor::test_support::NullEventLog);
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let services_arc = Arc::new(services);
    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(Arc::clone(&services_arc), event_log, registry, &config)
        .expect("reactor should construct");
    let reactor_arc = Arc::new(reactor);
    let handle = weft_reactor::services::ReactorHandle::new(Arc::clone(&reactor_arc));

    services_arc
        .reactor_handle
        .set(Arc::new(handle))
        .expect("OnceLock::set should succeed on first call");

    assert!(
        services_arc.reactor_handle.get().is_some(),
        "reactor_handle should be Some after OnceLock::set"
    );
}

#[test]
fn oncelock_second_set_returns_err() {
    let services = weft_reactor::test_support::make_test_services();
    let event_log: Arc<dyn weft_reactor::event_log::EventLog> =
        Arc::new(weft_reactor::test_support::NullEventLog);
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let services_arc = Arc::new(services);
    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(Arc::clone(&services_arc), event_log, registry, &config)
        .expect("reactor should construct");
    let reactor_arc = Arc::new(reactor);

    let handle1 = Arc::new(weft_reactor::services::ReactorHandle::new(Arc::clone(
        &reactor_arc,
    )));
    let handle2 = Arc::new(weft_reactor::services::ReactorHandle::new(Arc::clone(
        &reactor_arc,
    )));

    assert!(services_arc.reactor_handle.set(handle1).is_ok());
    assert!(
        services_arc.reactor_handle.set(handle2).is_err(),
        "second OnceLock::set should fail"
    );
}

#[test]
fn child_budget_at_max_depth_returns_err() {
    use weft_reactor::budget::{Budget, BudgetExhaustedReason};
    let deadline = chrono::Utc::now() + chrono::Duration::hours(1);
    let mut budget = Budget::new(10, 5, 3, deadline);
    budget.current_depth = 2;
    let result = budget.child_budget();
    assert!(
        matches!(result, Err(BudgetExhaustedReason::Depth)),
        "child_budget() should return Err(Depth) when at max_depth-1"
    );
}

#[tokio::test]
async fn spawn_child_returns_err_at_depth_limit() {
    let services = Arc::new(weft_reactor::test_support::make_test_services());
    let event_log: Arc<dyn weft_reactor::event_log::EventLog> =
        Arc::new(weft_reactor::test_support::NullEventLog);
    let registry = build_registry(vec![
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(Arc::clone(&services), event_log, registry, &config).unwrap();
    let reactor_arc = Arc::new(reactor);
    let handle = weft_reactor::services::ReactorHandle::new(Arc::clone(&reactor_arc));

    let deadline = chrono::Utc::now() + chrono::Duration::hours(1);
    let mut parent_budget = weft_reactor::budget::Budget::new(10, 5, 3, deadline);
    parent_budget.current_depth = 2;

    let (parent_tx, _parent_rx) = mpsc::channel::<PipelineEvent>(16);
    let result = handle
        .spawn_child(
            test_request(),
            TenantId("t1".to_string()),
            RequestId("r1".to_string()),
            weft_reactor::execution::ExecutionId::new(),
            parent_budget,
            parent_tx,
            None,
            "default",
        )
        .await;

    assert!(
        matches!(
            result,
            Err(weft_reactor::error::ReactorError::BudgetExhausted(_))
        ),
        "spawn_child should return BudgetExhausted when at depth limit, got: {:?}",
        result
    );
}

#[tokio::test]
async fn spawn_child_creates_child_with_correct_parent_id_and_depth() {
    let services =
        Arc::new(weft_reactor::test_support::make_test_services_with_response("child response"));
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(TextGenerateActivity {
            name: "generate".to_string(),
            response_text: "child response".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor =
        Reactor::new(Arc::clone(&services), event_log.clone(), registry, &config).unwrap();
    let reactor_arc = Arc::new(reactor);
    let handle = Arc::new(weft_reactor::services::ReactorHandle::new(Arc::clone(
        &reactor_arc,
    )));

    services
        .reactor_handle
        .set(Arc::clone(&handle))
        .expect("OnceLock::set should succeed");

    let parent_id = weft_reactor::execution::ExecutionId::new();
    let deadline = chrono::Utc::now() + chrono::Duration::hours(1);
    let parent_budget = weft_reactor::budget::Budget::new(10, 5, 3, deadline);

    let (parent_tx, mut parent_rx) = mpsc::channel::<PipelineEvent>(16);

    let result = handle
        .spawn_child(
            test_request(),
            TenantId("t1".to_string()),
            RequestId("r1".to_string()),
            parent_id.clone(),
            parent_budget.clone(),
            parent_tx,
            None,
            "default",
        )
        .await;

    assert!(result.is_ok(), "spawn_child should succeed: {:?}", result);

    let mut found_child_completed = false;
    while let Ok(event) = parent_rx.try_recv() {
        if matches!(
            event,
            PipelineEvent::Child(weft_reactor::ChildEvent::Completed { .. })
        ) {
            found_child_completed = true;
            break;
        }
    }
    assert!(
        found_child_completed,
        "ChildCompleted event should arrive on parent's channel"
    );

    let final_budget = result.unwrap();
    assert_eq!(
        final_budget.current_depth, 1,
        "child budget should have depth 1"
    );

    let all_execs = event_log.all_executions();
    let child_exec = all_execs
        .iter()
        .find(|e| e.id != parent_id)
        .expect("child execution record should exist in event log");
    assert_eq!(
        child_exec.parent_id.as_ref(),
        Some(&parent_id),
        "child execution record should have parent_id set to the calling parent"
    );
    assert_eq!(child_exec.depth, 1, "child execution should be at depth 1");
}

#[tokio::test]
async fn spawn_child_budget_deduction_works() {
    use weft_reactor::budget::Budget;

    let deadline = chrono::Utc::now() + chrono::Duration::hours(1);
    let parent_budget = Budget::new(10, 5, 3, deadline);

    let child_budget_initial = parent_budget.child_budget().unwrap();
    let mut child_budget_final = child_budget_initial.clone();
    child_budget_final.remaining_generation_calls = 7;

    let mut parent_after = parent_budget.clone();
    parent_after.deduct_child_usage(&child_budget_final);

    assert_eq!(
        parent_after.remaining_generation_calls, 7,
        "parent should have 7 gen calls remaining after child used 3"
    );
}

#[test]
fn cancellation_token_child_hierarchy_propagates() {
    let parent_cancel = CancellationToken::new();
    let child_cancel = parent_cancel.child_token();

    assert!(!parent_cancel.is_cancelled());
    assert!(!child_cancel.is_cancelled());

    parent_cancel.cancel();

    assert!(parent_cancel.is_cancelled());
    assert!(
        child_cancel.is_cancelled(),
        "cancelling parent should cancel child token"
    );
}

#[tokio::test]
async fn spawn_child_with_cancelled_parent_fails_or_cancels() {
    let services =
        Arc::new(weft_reactor::test_support::make_test_services_with_response("child response"));
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(TextGenerateActivity {
            name: "generate".to_string(),
            response_text: "child response".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);
    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor =
        Reactor::new(Arc::clone(&services), event_log.clone(), registry, &config).unwrap();
    let reactor_arc = Arc::new(reactor);
    let handle = weft_reactor::services::ReactorHandle::new(Arc::clone(&reactor_arc));

    let parent_cancel = CancellationToken::new();
    parent_cancel.cancel();

    let parent_id = weft_reactor::execution::ExecutionId::new();
    let deadline = chrono::Utc::now() + chrono::Duration::hours(1);
    let parent_budget = weft_reactor::budget::Budget::new(10, 5, 3, deadline);
    let (parent_tx, _parent_rx) = mpsc::channel::<PipelineEvent>(16);

    let result = handle
        .spawn_child(
            test_request(),
            TenantId("t1".to_string()),
            RequestId("r1".to_string()),
            parent_id,
            parent_budget,
            parent_tx,
            Some(&parent_cancel),
            "default",
        )
        .await;

    let _ = result;
}

// ── Command iteration loop ─────────────────────────────────────────────────

#[tokio::test]
async fn command_iteration_loop_executes_command_then_calls_generate_again() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let call_count_clone = Arc::clone(&call_count);

    struct CommandThenDoneActivity {
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    #[async_trait::async_trait]
    impl Activity for CommandThenDoneActivity {
        fn name(&self) -> &str {
            "generate"
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
            let call_n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Started {
                    model: "stub-model".to_string(),
                    message_count: input.messages.len(),
                }))
                .await;

            if call_n == 0 {
                let invocation = CommandInvocation {
                    name: "test_command".to_string(),
                    action: CommandAction::Execute,
                    arguments: serde_json::json!({"arg": "value"}),
                };
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::CommandInvocation(invocation),
                    )))
                    .await;
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
                    name: "generate".to_string(),
                    idempotency_key: input.idempotency_key.clone(),
                }))
                .await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(CommandThenDoneActivity {
            call_count: call_count_clone,
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("command iteration execution should succeed");

    assert_eq!(
        call_count.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "generate activity should be called twice (once for command, once after results)"
    );

    assert!(
        result.budget_used.iterations >= 1,
        "at least one iteration should be recorded"
    );

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();

    assert!(
        event_types.contains(&"command.completed"),
        "event log should contain command.completed; got: {event_types:?}"
    );
    assert!(
        event_types.contains(&"execution.iteration_completed"),
        "event log should contain execution.iteration_completed; got: {event_types:?}"
    );
}

// ── PreResponse hook retry ─────────────────────────────────────────────────

#[tokio::test]
async fn pre_response_hook_block_injects_feedback_and_retries_generation() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let gen_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let gen_calls_clone = Arc::clone(&gen_calls);
    let hook_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let hook_calls_clone = Arc::clone(&hook_calls);

    struct CountingDoneActivity {
        name: String,
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    #[async_trait::async_trait]
    impl Activity for CountingDoneActivity {
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
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name.clone(),
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

    struct BlockOnceThenAllowHook {
        name: String,
        call_count: Arc<std::sync::atomic::AtomicU32>,
        block_reason: String,
    }

    #[async_trait::async_trait]
    impl Activity for BlockOnceThenAllowHook {
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
            let call_n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name.clone(),
                }))
                .await;

            if call_n == 0 {
                let _ = event_tx
                    .send(PipelineEvent::Hook(HookOutcome::Blocked {
                        hook_event: "pre_response".to_string(),
                        hook_name: self.name.clone(),
                        reason: self.block_reason.clone(),
                    }))
                    .await;
            } else {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Completed {
                        name: self.name.clone(),
                        idempotency_key: input.idempotency_key.clone(),
                    }))
                    .await;
            }
        }
    }

    let pipeline_config = PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("generate".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks {
            pre_response: vec![ActivityRef::Name("pre_response_hook".to_string())],
            ..LoopHooks::default()
        },
    };

    let registry = build_registry(vec![
        Arc::new(CountingDoneActivity {
            name: "generate".to_string(),
            call_count: gen_calls_clone,
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
        Arc::new(BlockOnceThenAllowHook {
            name: "pre_response_hook".to_string(),
            call_count: hook_calls_clone,
            block_reason: "content policy violation: try again".to_string(),
        }),
    ]);

    let config = reactor_config(pipeline_config);
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        result.is_ok(),
        "pre_response hook retry should succeed on second generation; got: {:?}",
        result
    );

    assert_eq!(
        gen_calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "generate should be called twice: once per pre_response hook block+retry cycle"
    );

    assert_eq!(
        hook_calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "pre_response hook should be called twice"
    );

    let (exec_result, _) = result.unwrap();
    let events = event_log
        .read(&exec_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        event_types.contains(&"hook.blocked"),
        "event log should contain hook.blocked; got: {event_types:?}"
    );
}

// ── Retry + cancellation ───────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn cancel_during_retry_backoff_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    struct FailWithCancelActivity;

    #[async_trait::async_trait]
    impl Activity for FailWithCancelActivity {
        fn name(&self) -> &str {
            "generate"
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
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Signal(SignalEvent::Received(
                    Signal::Cancel {
                        reason: "cancel during backoff".to_string(),
                    },
                )))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "generate".to_string(),
                    error: "transient failure".to_string(),
                    retryable: true,
                }))
                .await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(FailWithCancelActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 3,
                    initial_backoff_ms: 5000,
                    max_backoff_ms: 10_000,
                    backoff_multiplier: 1.0,
                }),
                timeout_secs: None,
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let reactor_arc = Arc::new(reactor);
    let reactor_ref = Arc::clone(&reactor_arc);

    let handle = tokio::spawn(async move {
        reactor_ref
            .execute(
                test_request(),
                TenantId("tenant1".to_string()),
                RequestId("req1".to_string()),
                None,
                None,
                None,
                None,
            )
            .await
    });

    tokio::time::advance(Duration::from_millis(100)).await;

    let result = handle.await.expect("task should complete");

    assert!(
        result.is_ok(),
        "cancel during retry should return Ok (not Err): {:?}",
        result
    );

    let (exec_result, _) = result.unwrap();

    let events = event_log
        .read(&exec_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        event_types.contains(&"execution.cancelled"),
        "event log should contain execution.cancelled; got: {event_types:?}"
    );

    assert!(
        !event_types.contains(&"activity.retried"),
        "no activity.retried should appear when cancel fires before retry; got: {event_types:?}"
    );
}

// ── Per-chunk timeout reset ────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn per_chunk_timeout_resets_after_each_chunk() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    struct OneChunkThenStallActivity;

    #[async_trait::async_trait]
    impl Activity for OneChunkThenStallActivity {
        fn name(&self) -> &str {
            "generate"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Started {
                    model: "stub-model".to_string(),
                    message_count: 1,
                }))
                .await;

            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                    GeneratedEvent::Content {
                        part: ContentPart::Text("first chunk".to_string()),
                    },
                )))
                .await;

            cancel.cancelled().await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "generate".to_string(),
                    error: "cancelled".to_string(),
                    retryable: false,
                }))
                .await;
        }
    }

    let registry = build_registry(vec![
        Arc::new(OneChunkThenStallActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: None,
                timeout_secs: Some(2),
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 2,
            command_timeout_secs: 10,
        },
    };

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let reactor_arc = Arc::new(reactor);
    let reactor_ref = Arc::clone(&reactor_arc);

    let handle = tokio::spawn(async move {
        reactor_ref
            .execute(
                test_request(),
                TenantId("tenant1".to_string()),
                RequestId("req1".to_string()),
                None,
                None,
                None,
                None,
            )
            .await
    });

    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::time::advance(Duration::from_secs(2)).await;

    let result = handle.await.expect("task should complete");

    assert!(
        result.is_err(),
        "per-chunk timeout should fail execution after silence; got Ok"
    );
}

// ── should_retry / backoff_ms unit tests ─────────────────────────────────

#[test]
fn should_retry_false_when_no_policy() {
    let budget = Budget::new(10, 5, 3, chrono::Utc::now() + chrono::Duration::hours(1));
    let cancel = CancellationToken::new();
    let result = weft_reactor::reactor::test_hooks::should_retry_pub(None, 0, &budget, &cancel);
    assert!(!result, "should not retry without policy");
}

#[test]
fn should_retry_false_when_attempts_exhausted() {
    let policy = RetryPolicy {
        max_retries: 2,
        initial_backoff_ms: 100,
        max_backoff_ms: 1000,
        backoff_multiplier: 2.0,
    };
    let budget = Budget::new(10, 5, 3, chrono::Utc::now() + chrono::Duration::hours(1));
    let cancel = CancellationToken::new();
    let result =
        weft_reactor::reactor::test_hooks::should_retry_pub(Some(&policy), 2, &budget, &cancel);
    assert!(!result);
}

#[test]
fn should_retry_false_when_cancelled() {
    let policy = RetryPolicy {
        max_retries: 5,
        initial_backoff_ms: 100,
        max_backoff_ms: 1000,
        backoff_multiplier: 2.0,
    };
    let budget = Budget::new(10, 5, 3, chrono::Utc::now() + chrono::Duration::hours(1));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let result =
        weft_reactor::reactor::test_hooks::should_retry_pub(Some(&policy), 0, &budget, &cancel);
    assert!(!result, "should not retry when cancelled");
}

#[test]
fn backoff_ms_respects_cap() {
    let policy = RetryPolicy {
        max_retries: 10,
        initial_backoff_ms: 1000,
        max_backoff_ms: 5000,
        backoff_multiplier: 2.0,
    };
    let ms = weft_reactor::reactor::test_hooks::backoff_ms_pub(&policy, 5);
    assert!(ms >= 5000, "backoff should be at least max (before jitter)");
    assert!(
        ms < 5000 + 5000 / 4 + 2,
        "backoff should not exceed cap + 25% jitter"
    );
}

// ── Phase 5: Pre-loop activity wiring integration tests ───────────────────

use weft_reactor::activities::{
    CommandFormattingActivity, CommandSelectionActivity, ModelSelectionActivity,
    ProviderResolutionActivity, SamplingAdjustmentActivity, SystemPromptAssemblyActivity,
    ValidateActivity,
};

fn build_new_preloop_registry(generate_name: &str) -> Arc<ActivityRegistry> {
    build_registry(vec![
        Arc::new(ValidateActivity),
        Arc::new(ModelSelectionActivity),
        Arc::new(CommandSelectionActivity),
        Arc::new(ProviderResolutionActivity),
        Arc::new(SystemPromptAssemblyActivity),
        Arc::new(CommandFormattingActivity),
        Arc::new(SamplingAdjustmentActivity),
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

fn new_preloop_pipeline_config(generate_name: &str) -> PipelineConfig {
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

#[tokio::test]
async fn pre_loop_all_six_activities_produce_expected_events() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_new_preloop_registry("generate");
    let config = reactor_config(new_preloop_pipeline_config("generate"));

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();

    assert!(
        event_types.contains(&"selection.model_selected"),
        "missing selection.model_selected event; got: {event_types:?}"
    );
    assert!(
        event_types.contains(&"selection.commands_selected"),
        "missing selection.commands_selected event; got: {event_types:?}"
    );
    assert!(
        event_types.contains(&"selection.provider_resolved"),
        "missing selection.provider_resolved event; got: {event_types:?}"
    );
    assert!(
        event_types.contains(&"context.system_prompt_assembled"),
        "missing context.system_prompt_assembled event; got: {event_types:?}"
    );
    assert!(
        event_types.contains(&"context.commands_formatted"),
        "missing context.commands_formatted event; got: {event_types:?}"
    );
    assert!(
        event_types.contains(&"context.sampling_updated"),
        "missing context.sampling_updated event; got: {event_types:?}"
    );
}

#[tokio::test]
async fn pre_loop_events_appear_in_correct_order() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_new_preloop_registry("generate");
    let config = reactor_config(new_preloop_pipeline_config("generate"));

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();

    let seq_of = |event_type: &str| -> u64 {
        events
            .iter()
            .find(|e| e.event_type == event_type)
            .map(|e| e.sequence)
            .unwrap_or(0)
    };

    let model_seq = seq_of("selection.model_selected");
    let commands_seq = seq_of("selection.commands_selected");
    let provider_seq = seq_of("selection.provider_resolved");
    let sys_prompt_seq = seq_of("context.system_prompt_assembled");
    let cmd_fmt_seq = seq_of("context.commands_formatted");
    let sampling_seq = seq_of("context.sampling_updated");

    assert!(
        model_seq > 0,
        "selection.model_selected must be present (seq > 0)"
    );
    assert!(
        model_seq < commands_seq,
        "selection.model_selected ({model_seq}) must precede selection.commands_selected ({commands_seq})"
    );
    assert!(
        commands_seq < provider_seq,
        "selection.commands_selected ({commands_seq}) must precede selection.provider_resolved ({provider_seq})"
    );
    assert!(
        provider_seq < sys_prompt_seq,
        "selection.provider_resolved ({provider_seq}) must precede context.system_prompt_assembled ({sys_prompt_seq})"
    );
    assert!(
        sys_prompt_seq < cmd_fmt_seq,
        "context.system_prompt_assembled ({sys_prompt_seq}) must precede context.commands_formatted ({cmd_fmt_seq})"
    );
    assert!(
        cmd_fmt_seq < sampling_seq,
        "context.commands_formatted ({cmd_fmt_seq}) must precede context.sampling_updated ({sampling_seq})"
    );
}

#[tokio::test]
async fn pre_loop_system_prompt_at_index_zero() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_new_preloop_registry("generate");
    let config = reactor_config(new_preloop_pipeline_config("generate"));

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();

    let system_prompt_injected = events.iter().any(|e| {
        if e.event_type != "context.message_injected" {
            return false;
        }
        e.payload
            .get("event")
            .and_then(|v| v.get("source"))
            .and_then(|s| s.as_str())
            .map(|s| s == "SystemPromptAssembly")
            .unwrap_or(false)
    });
    assert!(
        system_prompt_injected,
        "expected context.message_injected with SystemPromptAssembly source in event log"
    );

    let has_sys_prompt = events
        .iter()
        .any(|e| e.event_type == "context.system_prompt_assembled");
    assert!(
        has_sys_prompt,
        "expected context.system_prompt_assembled event in event log"
    );
}

#[tokio::test]
async fn pre_loop_generation_config_includes_model() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_new_preloop_registry("generate");
    let config = reactor_config(new_preloop_pipeline_config("generate"));

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("execution should succeed");

    let events = event_log
        .read(&result.execution_id, None::<u64>)
        .await
        .unwrap();

    let model_selected = events
        .iter()
        .find(|e| e.event_type == "selection.model_selected")
        .expect("selection.model_selected must be in event log");

    let model_name = model_selected
        .payload
        .get("event")
        .and_then(|v| v.get("model_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        !model_name.is_empty(),
        "model_name in selection.model_selected must be non-empty"
    );

    let sampling_updated = events
        .iter()
        .find(|e| e.event_type == "context.sampling_updated")
        .expect("context.sampling_updated must be in event log");

    let max_tokens = sampling_updated
        .payload
        .get("event")
        .and_then(|v| v.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        max_tokens > 0,
        "max_tokens in context.sampling_updated must be > 0, got {max_tokens}"
    );

    assert!(
        events.iter().any(|e| e.event_type == "execution.completed"),
        "execution must complete successfully"
    );
}

#[tokio::test]
async fn pre_loop_activity_failure_terminates_execution() {
    struct FailingModelSelection;

    #[async_trait::async_trait]
    impl Activity for FailingModelSelection {
        fn name(&self) -> &str {
            "model_selection"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn weft_reactor::event_log::EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name().to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "model_selection: no eligible models".to_string(),
                    retryable: false,
                }))
                .await;
        }
    }

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(ValidateActivity),
        Arc::new(FailingModelSelection),
        Arc::new(CommandSelectionActivity),
        Arc::new(ProviderResolutionActivity),
        Arc::new(SystemPromptAssemblyActivity),
        Arc::new(CommandFormattingActivity),
        Arc::new(SamplingAdjustmentActivity),
        Arc::new(ImmediateDoneActivity {
            name: "generate".to_string(),
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(new_preloop_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            test_request(),
            TenantId("tenant1".to_string()),
            RequestId("req1".to_string()),
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        result.is_err(),
        "execution should fail when model_selection activity fails"
    );
}
