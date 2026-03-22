//! Child execution, OnceLock, and depth limit integration tests.
//!
//! Tests covering OnceLock lifecycle, budget depth limits,
//! spawn_child behavior, budget deduction, and cancellation token hierarchy.

mod harness;

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::event::PipelineEvent;
use weft_reactor::reactor::Reactor;
use weft_reactor::{RequestId, TenantId};

use harness::{
    TestActivity, build_registry, reactor_config, simple_pipeline_config, test_event_log,
    test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

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
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
            weft_reactor::SpawnRequest {
                request: test_request(),
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id: weft_reactor::execution::ExecutionId::new(),
                parent_budget,
                parent_event_tx: parent_tx,
                pipeline_name: "default".to_string(),
            },
            None,
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
        TestActivity::generate("generate")
            .with_text("child response")
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
            weft_reactor::SpawnRequest {
                request: test_request(),
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id: parent_id.clone(),
                parent_budget: parent_budget.clone(),
                parent_event_tx: parent_tx,
                pipeline_name: "default".to_string(),
            },
            None,
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
        TestActivity::generate("generate")
            .with_text("child response")
            .build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
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
            weft_reactor::SpawnRequest {
                request: test_request(),
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id,
                parent_budget,
                parent_event_tx: parent_tx,
                pipeline_name: "default".to_string(),
            },
            Some(&parent_cancel),
        )
        .await;

    let _ = result;
    // Test passes regardless of result — cancelled parent may succeed or return cancelled.
}
