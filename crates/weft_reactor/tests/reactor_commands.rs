//! Command iteration loop and failure isolation integration tests.
//!
//! Tests covering the command execution loop, command failure injection,
//! command timeouts, multiple command failures, and degradation on success.

mod harness;

use std::sync::Arc;
use std::time::Duration;

use weft_reactor::event::FailureDetail;
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use harness::{
    CallAction, EventAssertions, TestActivity, build_registry, failing_execute_command,
    hanging_execute_command, reactor_config, simple_pipeline_config, test_event_log, test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

#[tokio::test]
async fn command_iteration_loop_executes_command_then_calls_generate_again() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let registry = build_registry(vec![
        TestActivity::generate("generate")
            .invokes_commands(vec!["test_command"])
            .with_call_count(Arc::clone(&call_count))
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

    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .contains("command.completed")
        .contains("execution.iteration_completed");
}

#[tokio::test]
async fn command_failure_injects_error_message_and_continues() {
    use pretty_assertions::assert_eq;

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        // First call invokes a command; second call returns Done.
        TestActivity::generate("generate")
            .invokes_commands(vec!["failing_tool"])
            .with_call_count(Arc::clone(&call_count))
            .build(),
        TestActivity::assemble_response().into(),
        failing_execute_command(
            "command not found: failing_tool",
            FailureDetail {
                error_code: "command_not_found".to_string(),
                detail: serde_json::json!({ "command_name": "failing_tool" }),
                cause: Some("command not found: failing_tool".to_string()),
                attempted: Some("execute command failing_tool".to_string()),
                fallback: None,
            },
        ),
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
    // Source must be CommandError: payload_contains cannot express "key exists" without a known
    // value, so use all() as the escape hatch to check for CommandError presence.
    let assertions = EventAssertions::for_execution(&event_log, &execution_result.execution_id)
        .await
        .contains("context.message_injected");
    let has_command_error = assertions.all().iter().any(|e| {
        e.event_type == "context.message_injected"
            && e.payload
                .get("event")
                .and_then(|v| v.get("source"))
                .and_then(|s| s.get("CommandError"))
                .is_some()
    });
    assert!(
        has_command_error,
        "expected context.message_injected with CommandError source; events: {:?}",
        assertions.types()
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

    // Invoke slow_tool on first call; on subsequent calls check for injected error before invoking.
    let registry = build_registry(vec![
        Arc::new(harness::TestActivity {
            activity_name: "generate".to_string(),
            criticality: weft_reactor_trait::Criticality::Critical,
            behavior: harness::Behavior::PerCall {
                call_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
                actions: vec![CallAction::WithCommands {
                    commands: vec!["slow_tool".to_string()],
                }],
                default_action: CallAction::UnlessErrorSeen,
            },
        }),
        TestActivity::assemble_response().into(),
        hanging_execute_command(),
    ]);

    // Use the default reactor_config (command_timeout_secs = 10).
    let config = reactor_config(simple_pipeline_config("generate"));

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

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

    let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

    // InvokeTwoCommandsActivity: first call invokes two commands, subsequent calls return Done.
    let registry = build_registry(vec![
        Arc::new(harness::TestActivity {
            activity_name: "generate".to_string(),
            criticality: weft_reactor_trait::Criticality::Critical,
            behavior: harness::Behavior::PerCall {
                call_count: Arc::clone(&call_count),
                actions: vec![CallAction::WithCommands {
                    commands: vec!["tool_a".to_string(), "tool_b".to_string()],
                }],
                default_action: CallAction::Done,
            },
        }),
        TestActivity::assemble_response().into(),
        // Always-failing execute_command (NonCritical). Error message is static here;
        // the test validates error count and CommandError injection, not the message content.
        failing_execute_command(
            "command not found",
            FailureDetail {
                error_code: "command_not_found".to_string(),
                detail: serde_json::Value::Null,
                cause: Some("command not found".to_string()),
                attempted: None,
                fallback: None,
            },
        ),
    ]);

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();

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

    // Both error messages should appear in the event log as MessageInjected events with
    // CommandError source. Use all() to count CommandError injections (payload_contains
    // cannot express count-with-key-exists without knowing the exact value).
    let assertions = EventAssertions::for_execution(&event_log, &execution_result.execution_id)
        .await;
    let injected_count = assertions
        .all()
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
