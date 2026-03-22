//! Reactor lifecycle integration tests.
//!
//! Tests covering the happy path, budget exhaustion, cancellation,
//! timeouts, heartbeat monitoring, streaming, and event log completeness.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
use weft_reactor::event::{
    ActivityEvent, FailureDetail, GeneratedEvent, GenerationEvent, PipelineEvent, SignalEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::reactor::Reactor;
use weft_reactor::signal::Signal;
use weft_reactor::test_support::{make_test_services, make_test_services_with_response};
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use weft_core::ContentPart;

use harness::{
    ImmediateDoneActivity, StubAssembleResponse, StubExecuteCommand, TextGenerateActivity,
    build_registry, reactor_config, simple_pipeline_config, test_event_log, test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

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
        .await;

    assert!(
        result.is_ok(),
        "budget exhaustion should return Ok with partial results, got: {:?}",
        result
    );
    let (execution_result, _) = result.unwrap();
    assert_eq!(execution_result.budget_used.generation_calls, 1);
}

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
                    detail: FailureDetail::default(),
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
    });

    tokio::time::advance(Duration::from_secs(6)).await;

    let result = handle.await.expect("task should complete");

    assert!(
        result.is_err(),
        "generation timeout should fail execution without retry"
    );
}

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
                    detail: FailureDetail::default(),
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
    });

    tokio::time::advance(Duration::from_secs(10)).await;

    let result = handle.await.expect("task should complete");
    assert!(
        result.is_err(),
        "heartbeat miss should fail execution without retry: {:?}",
        result
    );
}

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
            ExecutionContext {
                request: test_request(),
                tenant_id: TenantId("tenant1".to_string()),
                request_id: RequestId("req1".to_string()),
                parent_id: None,
                parent_budget: None,
                client_tx: Some(client_tx),
            },
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
