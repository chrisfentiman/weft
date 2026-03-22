//! Reactor retry mechanics integration tests.
//!
//! Tests covering retry on failure, non-retryable errors, retry exhaustion,
//! budget-gated retry, cancel-during-backoff, and per-chunk timeout reset.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use weft_reactor::budget::Budget;
use weft_reactor::config::{
    ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig, RetryPolicy,
};
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{EventLog, ExecutionContext, RequestId, TenantId};

use harness::{EventAssertions, TestActivity, build_registry, test_event_log, test_request};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

#[tokio::test]
async fn generate_fails_once_then_succeeds_with_retry() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        TestActivity::generate("generate").fails_then_succeeds(1),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
    EventAssertions::for_execution(&event_log, &exec_id)
        .await
        .contains("activity.retried");
}

#[tokio::test]
async fn generate_not_retried_when_retryable_false() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        TestActivity::failing("generate")
            .with_error("auth error")
            .retryable(false)
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
        TestActivity::failing("generate")
            .with_error("transient error")
            .retryable(true)
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
        TestActivity::failing("generate")
            .with_error("transient error")
            .retryable(true)
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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

    EventAssertions::for_execution(&event_log, &exec_result.execution_id)
        .await
        .contains("budget.exhausted")
        .does_not_contain("activity.retried");
}

#[tokio::test(start_paused = true)]
async fn cancel_during_retry_backoff_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    // Activity that emits cancel signal then fails with retryable=true.
    // The cancel fires before the retry backoff can complete.
    use std::sync::Arc as StdArc;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_reactor::activity::{Activity, ActivityInput};
    use weft_reactor::event::{ActivityEvent, FailureDetail, PipelineEvent as PE, SignalEvent};
    use weft_reactor::execution::ExecutionId;
    use weft_reactor::signal::Signal;

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
            event_tx: mpsc::Sender<PE>,
            _cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PE::Activity(ActivityEvent::Started {
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PE::Signal(SignalEvent::Received(Signal::Cancel {
                    reason: "cancel during backoff".to_string(),
                })))
                .await;
            let _ = event_tx
                .send(PE::Activity(ActivityEvent::Failed {
                    name: "generate".to_string(),
                    error: "transient failure".to_string(),
                    retryable: true,
                    detail: FailureDetail::default(),
                }))
                .await;
        }
    }

    let registry = build_registry(vec![
        StdArc::new(FailWithCancelActivity),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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

    let reactor_arc = StdArc::new(reactor);
    let reactor_ref = StdArc::clone(&reactor_arc);

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

    EventAssertions::for_execution(&event_log, &exec_result.execution_id)
        .await
        .contains("execution.cancelled")
        .does_not_contain("activity.retried");
}

#[tokio::test(start_paused = true)]
async fn per_chunk_timeout_resets_after_each_chunk() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        TestActivity::stalling_after_chunk("generate", "first chunk").into(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
