//! Reactor retry mechanics integration tests.
//!
//! Tests covering retry on failure, non-retryable errors, retry exhaustion,
//! budget-gated retry, cancel-during-backoff, and per-chunk timeout reset.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::budget::Budget;
use weft_reactor::config::{
    ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig, RetryPolicy,
};
use weft_reactor::event::{
    ActivityEvent, FailureDetail, GeneratedEvent, GenerationEvent, PipelineEvent, SignalEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::reactor::Reactor;
use weft_reactor::signal::Signal;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use weft_core::ContentPart;

use harness::{
    AlwaysFailActivity, FailThenSucceedActivity, StubAssembleResponse, StubExecuteCommand,
    build_registry, test_event_log, test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

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
            ExecutionContext {
                request: test_request(),
                tenant_id: TenantId("tenant1".to_string()),
                request_id: RequestId("req1".to_string()),
                parent_id: None,
                parent_budget: Some(exhausted_budget),
                client_tx: None,
            },
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
                    detail: FailureDetail::default(),
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
                    detail: FailureDetail::default(),
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

    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::time::advance(Duration::from_secs(2)).await;

    let result = handle.await.expect("task should complete");

    assert!(
        result.is_err(),
        "per-chunk timeout should fail execution after silence; got Ok"
    );
}
