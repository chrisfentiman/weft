//! Hook blocking, pre-response retry, and pre-generate hook integration tests.
//!
//! Tests covering hook blocking in pre-loop, pre-response hook block+retry,
//! pre-generate hook non-critical failure, critical failure, and blocked.

mod harness;

use std::sync::Arc;

use weft_reactor::config::{ActivityRef, LoopHooks, PipelineConfig};
use weft_reactor::error::ReactorError;
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::{make_test_services, make_test_services_with_blocking_hook};
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use weft_core::HookEvent;
use weft_reactor_trait::Criticality;

use harness::{
    EventAssertions, TestActivity, build_registry, reactor_config, test_event_log, test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

#[tokio::test]
async fn hook_block_in_pre_loop_returns_hook_blocked_error() {
    let services = Arc::new(make_test_services_with_blocking_hook(
        HookEvent::RequestStart,
        "blocked by policy",
    ));
    let event_log = test_event_log();

    // HookStartActivity: runs hook chain, emits Blocked or Completed based on result.
    let registry = build_registry(vec![
        Arc::new(harness::TestActivity {
            activity_name: "hook_request_start".to_string(),
            criticality: Criticality::Critical,
            behavior: harness::Behavior::RunHook {
                hook_event: HookEvent::RequestStart,
                hook_event_name: "request_start".to_string(),
            },
        }),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
        matches!(result, Err(ReactorError::HookBlocked { .. })),
        "expected HookBlocked error, got: {:?}",
        result
    );
}

#[tokio::test]
async fn pre_response_hook_block_injects_feedback_and_retries_generation() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let gen_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let hook_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));

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
        // CountingDoneActivity: increments call_count, emits Generate Done.
        Arc::new(harness::TestActivity {
            activity_name: "generate".to_string(),
            criticality: Criticality::Critical,
            behavior: harness::Behavior::PerCall {
                call_count: Arc::clone(&gen_calls),
                actions: vec![],
                default_action: harness::CallAction::Done,
            },
        }),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
        // BlockOnceThenAllowHook: blocks on first call, allows on subsequent calls.
        Arc::new(harness::TestActivity {
            activity_name: "pre_response_hook".to_string(),
            criticality: Criticality::Critical,
            behavior: harness::Behavior::HookBlockOnce {
                hook_event: "pre_response".to_string(),
                hook_name: "pre_response_hook".to_string(),
                block_reason: "content policy violation: try again".to_string(),
                call_count: Arc::clone(&hook_calls),
            },
        }),
    ]);

    let config = reactor_config(pipeline_config);
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
    EventAssertions::for_execution(&event_log, &exec_result.execution_id)
        .await
        .contains("hook.blocked");
}

#[tokio::test]
async fn pre_generate_hook_non_critical_failure_degrades_and_continues() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        // NonCriticalFailingHookActivity: NonCritical, always emits ActivityEvent::Failed.
        TestActivity::failing("hook_pre_generate")
            .with_criticality(Criticality::NonCritical)
            .with_error("hook runner unavailable")
            .with_detail(weft_reactor::event::FailureDetail {
                error_code: "hook_runner_error".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("generate".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks {
            pre_generate: vec![ActivityRef::Name("hook_pre_generate".to_string())],
            ..LoopHooks::default()
        },
    });

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

    // Non-critical hook failure must not terminate execution.
    let (exec_result, _) = result
        .expect("non-critical pre_generate hook failure should degrade, not terminate execution");

    // ExecutionEvent::Degraded must be recorded for the hook failure,
    // and must reference hook_pre_generate as the activity that degraded.
    EventAssertions::for_execution(&event_log, &exec_result.execution_id)
        .await
        .payload_contains(
            "execution.degraded",
            "/event/notice/activity_name",
            &serde_json::json!("hook_pre_generate"),
        );
}

#[tokio::test]
async fn pre_generate_hook_critical_failure_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        // CriticalFailingHookActivity: Critical (default), always emits ActivityEvent::Failed.
        TestActivity::failing("hook_pre_generate")
            .with_error("critical hook failure")
            .with_detail(weft_reactor::event::FailureDetail {
                error_code: "hook_critical_error".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("generate".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks {
            pre_generate: vec![ActivityRef::Name("hook_pre_generate".to_string())],
            ..LoopHooks::default()
        },
    });

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

    assert!(
        result.is_err(),
        "critical pre_generate hook failure should terminate execution with an error"
    );
}

#[tokio::test]
async fn pre_generate_hook_blocked_terminates_regardless_of_criticality() {
    // BlockingNonCriticalHookActivity: NonCritical, emits HookOutcome::Blocked.
    // Uses NonCritical to confirm that criticality is irrelevant for blocked events.
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(harness::TestActivity {
            activity_name: "hook_pre_generate".to_string(),
            criticality: Criticality::NonCritical,
            behavior: harness::Behavior::HookBlockOnce {
                hook_event: "pre_generate".to_string(),
                hook_name: "policy_guard".to_string(),
                block_reason: "blocked by pre_generate policy".to_string(),
                // Always block (call_count never reaches 1 because test ends on first block).
                call_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            },
        }),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("generate".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks {
            pre_generate: vec![ActivityRef::Name("hook_pre_generate".to_string())],
            ..LoopHooks::default()
        },
    });

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

    assert!(
        matches!(result, Err(ReactorError::HookBlocked { .. })),
        "HookOutcome::Blocked from pre_generate hook should terminate execution with HookBlocked error; got: {result:?}"
    );
}
