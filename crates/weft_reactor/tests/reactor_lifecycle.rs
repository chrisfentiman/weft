//! Reactor lifecycle integration tests.
//!
//! Tests covering the happy path, budget exhaustion, cancellation,
//! timeouts, heartbeat monitoring, streaming, and event log completeness.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use weft_reactor::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
use weft_reactor::event::{GeneratedEvent, GenerationEvent, PipelineEvent};
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::{make_test_services, make_test_services_with_response};
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use tokio::sync::mpsc;
use weft_core::ContentPart;

use harness::{
    EventAssertions, TestActivity, build_registry, reactor_config, simple_pipeline_config,
    test_event_log, test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

#[tokio::test]
async fn simple_request_response_completes() {
    let services = Arc::new(make_test_services_with_response("Hello, world!"));
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::generate("generate")
            .with_text("Hello, world!")
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .contains("execution.started")
        .contains("execution.completed");
}

#[tokio::test]
async fn budget_exhaustion_terminates_gracefully() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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

    let registry = build_registry(vec![
        TestActivity::cancelling("generate", "test cancel via channel").into(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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

    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .contains("execution.cancelled");
}

/// Cancellation via Signal::Cancel sent on the event channel.
#[tokio::test]
async fn cancel_signal_on_channel_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        TestActivity::cancelling_with_sleep("generate", "test cancel", Duration::from_millis(100))
            .into(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
    EventAssertions::for_execution(&event_log, &exec_result.execution_id)
        .await
        .contains("execution.cancelled");
}

#[tokio::test(start_paused = true)]
async fn generation_timeout_fires_after_silence() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let registry = build_registry(vec![
        TestActivity::stalling("generate").into(),
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

    let registry = build_registry(vec![
        TestActivity::stalling_with_heartbeats("generate", 2, Duration::from_secs(1)).into(),
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
        TestActivity::generate("generate")
            .with_text("streaming token")
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
        TestActivity::generate("generate")
            .with_text("test response")
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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

    let assertions = EventAssertions::for_execution(&event_log, &result.execution_id).await;

    // Verify sequences are strictly increasing (requires raw events before consuming by value).
    let seqs: Vec<u64> = assertions.all().iter().map(|e| e.sequence).collect();
    for w in seqs.windows(2) {
        assert!(w[0] < w[1], "event sequences should be strictly increasing");
    }

    assertions
        .contains("execution.started")
        .contains("generation.started")
        .contains("generation.chunk")
        .contains("execution.completed");
}
