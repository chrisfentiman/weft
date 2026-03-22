//! Command iteration loop and failure isolation integration tests.
//!
//! Tests covering the command execution loop, command failure injection,
//! command timeouts, multiple command failures, and degradation on success.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::event::{
    ActivityEvent, FailureDetail, GeneratedEvent, GenerationEvent, PipelineEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use weft_core::{CommandAction, CommandInvocation, ContentPart, Role, Source, WeftMessage};

use harness::{
    StubAssembleResponse, build_registry, reactor_config, simple_pipeline_config, test_event_log,
    test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

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
        Arc::new(harness::StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let (result, _) = reactor
        .execute(
            ExecutionContext {
                request: test_request(),
                tenant_id: TenantId("tenant1".to_string()),
                request_id: RequestId("req1".to_string()),
                parent_id: None,
                parent_budget: None,
                client_tx: None,
            },
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

#[tokio::test]
async fn command_failure_injects_error_message_and_continues() {
    use pretty_assertions::assert_eq;

    /// Activity that invokes a command on the first generate call, then Done.
    struct InvokeCommandActivity {
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    #[async_trait::async_trait]
    impl Activity for InvokeCommandActivity {
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
                // First call: request a command invocation.
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::CommandInvocation(CommandInvocation {
                            name: "failing_tool".to_string(),
                            action: CommandAction::Execute,
                            arguments: serde_json::json!({}),
                        }),
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

    /// Activity that always pushes `ActivityEvent::Failed` (simulates infrastructure error).
    struct FailingExecuteCommandActivity;

    #[async_trait::async_trait]
    impl Activity for FailingExecuteCommandActivity {
        fn name(&self) -> &str {
            "execute_command"
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
                    name: "execute_command".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "execute_command".to_string(),
                    error: "command not found: failing_tool".to_string(),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "command_not_found".to_string(),
                        detail: serde_json::json!({ "command_name": "failing_tool" }),
                        cause: Some("command not found: failing_tool".to_string()),
                        attempted: Some("execute command failing_tool".to_string()),
                        fallback: None,
                    },
                }))
                .await;
        }
    }

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(InvokeCommandActivity {
            call_count: call_count_clone,
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(FailingExecuteCommandActivity),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            ExecutionContext {
                request: test_request(),
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id: None,
                parent_budget: None,
                client_tx: None,
            },
            None,
        )
        .await;

    let (execution_result, _) = result.expect(
        "command ActivityEvent::Failed should not kill the request; execution should complete Ok",
    );

    // The reactor injects an error message and records a DegradationNotice.
    assert_eq!(
        execution_result.degradations.len(),
        1,
        "should have one DegradationNotice from the failed execute_command"
    );
    assert_eq!(
        execution_result.degradations[0].activity_name, "execute_command",
        "degradation should be for execute_command"
    );
    assert_eq!(
        execution_result.degradations[0].error_code, "execution_error",
        "reactor records execution_error when execute_command fails"
    );

    // The error message must be injected into the event log so the LLM sees it on the next call.
    let events = event_log
        .read(&execution_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let injected = events.iter().any(|e| {
        if e.event_type != "context.message_injected" {
            return false;
        }
        // Source must be CommandError.
        e.payload
            .get("event")
            .and_then(|v| v.get("source"))
            .and_then(|s| s.get("CommandError"))
            .is_some()
    });
    assert!(
        injected,
        "expected context.message_injected with CommandError source in event log"
    );

    // Generate was called twice: once to request the command, once after error injection.
    assert_eq!(
        call_count.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "generate should be called twice: once before command, once after error injection"
    );
}

#[tokio::test(start_paused = true)]
async fn command_timeout_injects_error_message_and_continues() {
    use pretty_assertions::assert_eq;

    /// Generate that invokes a command once, then Done.
    struct InvokeOnceActivity;

    #[async_trait::async_trait]
    impl Activity for InvokeOnceActivity {
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

            // Only request command on first call. On the second call just return Done.
            // We detect "second call" by checking if messages contain the injected error.
            let already_has_error = input.messages.iter().any(|m| {
                m.content
                    .iter()
                    .any(|c| matches!(c, ContentPart::Text(t) if t.contains("failed")))
            });

            if !already_has_error {
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::CommandInvocation(CommandInvocation {
                            name: "slow_tool".to_string(),
                            action: CommandAction::Execute,
                            arguments: serde_json::json!({}),
                        }),
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

    /// Activity that hangs forever (simulating a slow command — will be timed out).
    struct HangingExecuteCommandActivity;

    #[async_trait::async_trait]
    impl Activity for HangingExecuteCommandActivity {
        fn name(&self) -> &str {
            "execute_command"
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
            _event_tx: mpsc::Sender<PipelineEvent>,
            cancel: CancellationToken,
        ) {
            // Hang until cancelled (simulates a command that never responds).
            cancel.cancelled().await;
        }
    }

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(InvokeOnceActivity),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(HangingExecuteCommandActivity),
    ]);

    // Use the default reactor_config (command_timeout_secs = 10).
    let config = reactor_config(simple_pipeline_config("generate"));

    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    // Drive the reactor to completion while advancing simulated time past the 10-second deadline.
    let reactor_fut = reactor.execute(
        ExecutionContext {
            request: test_request(),
            tenant_id: TenantId("t1".to_string()),
            request_id: RequestId("r1".to_string()),
            parent_id: None,
            parent_budget: None,
            client_tx: None,
        },
        None,
    );

    // Advance time past the 10-second command deadline.
    let result = tokio::time::timeout(Duration::from_secs(30), async {
        let advance_handle = tokio::spawn(async {
            // Give the reactor a moment to get into the command wait loop.
            tokio::time::sleep(Duration::from_millis(10)).await;
            tokio::time::advance(Duration::from_secs(15)).await;
        });
        let r = reactor_fut.await;
        let _ = advance_handle.await;
        r
    })
    .await
    .expect("test should not time out");

    let (execution_result, _) =
        result.expect("command timeout should not kill the request; execution should complete Ok");

    assert_eq!(
        execution_result.degradations.len(),
        1,
        "should have one DegradationNotice from the timed-out execute_command"
    );
    assert_eq!(
        execution_result.degradations[0].activity_name, "execute_command",
        "degradation should be for execute_command"
    );
    assert!(
        execution_result.degradations[0]
            .message
            .contains("timed out"),
        "degradation message should mention timeout; got: {}",
        execution_result.degradations[0].message
    );
}

#[tokio::test]
async fn multiple_command_failures_all_inject_errors_and_continue() {
    use pretty_assertions::assert_eq;

    /// Generate that invokes two commands on first call, then Done.
    struct InvokeTwoCommandsActivity {
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    #[async_trait::async_trait]
    impl Activity for InvokeTwoCommandsActivity {
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
                // First call: invoke two commands.
                for cmd in &["tool_a", "tool_b"] {
                    let _ = event_tx
                        .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                            GeneratedEvent::CommandInvocation(CommandInvocation {
                                name: cmd.to_string(),
                                action: CommandAction::Execute,
                                arguments: serde_json::json!({}),
                            }),
                        )))
                        .await;
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
                    name: "generate".to_string(),
                    idempotency_key: input.idempotency_key.clone(),
                }))
                .await;
        }
    }

    /// Activity that always fails with ActivityEvent::Failed.
    struct AlwaysFailingExecuteCommandActivity;

    #[async_trait::async_trait]
    impl Activity for AlwaysFailingExecuteCommandActivity {
        fn name(&self) -> &str {
            "execute_command"
        }

        fn criticality(&self) -> weft_reactor_trait::Criticality {
            weft_reactor_trait::Criticality::NonCritical
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
            // Extract the command name from metadata so we can echo it back in the error.
            let cmd_name = input
                .metadata
                .get("invocation")
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "execute_command".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "execute_command".to_string(),
                    error: format!("command not found: {cmd_name}"),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "command_not_found".to_string(),
                        detail: serde_json::json!({ "command_name": cmd_name }),
                        cause: Some(format!("command not found: {cmd_name}")),
                        attempted: Some(format!("execute command {cmd_name}")),
                        fallback: None,
                    },
                }))
                .await;
        }
    }

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(InvokeTwoCommandsActivity {
            call_count: call_count_clone,
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(AlwaysFailingExecuteCommandActivity),
    ]);

    let config = reactor_config(simple_pipeline_config("generate"));
    let reactor = Reactor::new(services, event_log.clone(), registry, &config)
        .expect("reactor should construct");

    let result = reactor
        .execute(
            ExecutionContext {
                request: test_request(),
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id: None,
                parent_budget: None,
                client_tx: None,
            },
            None,
        )
        .await;

    let (execution_result, _) = result
        .expect("all-commands-fail should not kill the request; execution should complete Ok");

    // Both commands should produce a degradation notice.
    assert_eq!(
        execution_result.degradations.len(),
        2,
        "should have two DegradationNotices, one per failed command"
    );
    for notice in &execution_result.degradations {
        assert_eq!(
            notice.activity_name, "execute_command",
            "each degradation should be for execute_command"
        );
        assert_eq!(
            notice.error_code, "execution_error",
            "reactor records execution_error for each failed command"
        );
    }

    // Both error messages should appear in the event log as MessageInjected events.
    let events = event_log
        .read(&execution_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let injected_count = events
        .iter()
        .filter(|e| {
            e.event_type == "context.message_injected"
                && e.payload
                    .get("event")
                    .and_then(|v| v.get("source"))
                    .and_then(|s| s.get("CommandError"))
                    .is_some()
        })
        .count();
    assert_eq!(
        injected_count, 2,
        "expected two context.message_injected(CommandError) events, one per failed command; got {injected_count}"
    );

    // Generate was called twice: once for the two commands, once after error injection.
    assert_eq!(
        call_count.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "generate should be called twice: once requesting commands, once after error injection"
    );
}

