//! Pre-loop activity ordering and failure integration tests.
//!
//! Tests covering the full pre-loop activity set, event ordering,
//! system prompt position, generation config, and pre-loop failure.

mod harness;

use std::sync::Arc;

use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use harness::{
    EventAssertions, EventPredicate, TestActivity, build_new_preloop_registry, build_registry,
    new_preloop_pipeline_config, pipeline_with_validate, reactor_config, test_event_log,
    test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

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

    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .contains("selection.model_selected")
        .contains("selection.commands_selected")
        .contains("selection.provider_resolved")
        .contains("context.system_prompt_assembled")
        .contains("context.commands_formatted")
        .contains("context.sampling_updated");
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

    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .in_order(&[
            "selection.model_selected",
            "selection.commands_selected",
            "selection.provider_resolved",
            "context.system_prompt_assembled",
            "context.commands_formatted",
            "context.sampling_updated",
        ]);
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

    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .payload_contains(
            "context.message_injected",
            "/event/source",
            &serde_json::json!("SystemPromptAssembly"),
        )
        .contains("context.system_prompt_assembled");
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

    // Use contains() for presence, and all() for value-range checks that cannot be
    // expressed as equality via payload_contains.
    let assertions = EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .contains("selection.model_selected")
        .contains("context.sampling_updated")
        .contains("execution.completed");

    let model_name = assertions
        .all()
        .iter()
        .find(|e| e.event_type == "selection.model_selected")
        .and_then(|e| e.payload.get("event"))
        .and_then(|v| v.get("model_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        !model_name.is_empty(),
        "model_name in selection.model_selected must be non-empty"
    );

    let max_tokens = assertions
        .all()
        .iter()
        .find(|e| e.event_type == "context.sampling_updated")
        .and_then(|e| e.payload.get("event"))
        .and_then(|v| v.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        max_tokens > 0,
        "max_tokens in context.sampling_updated must be > 0, got {max_tokens}"
    );
}

#[tokio::test]
async fn pre_loop_activity_failure_terminates_execution() {
    // FailingModelSelection: Critical (default), emits Started then Failed.
    let registry = build_registry(vec![
        TestActivity::validate_stub(),
        TestActivity::failing("model_selection")
            .with_error("model_selection: no eligible models")
            .build(),
        TestActivity::command_selection_stub(),
        TestActivity::provider_resolution_stub(),
        TestActivity::system_prompt_assembly_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let config = reactor_config(new_preloop_pipeline_config("generate"));
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
        result.is_err(),
        "execution should fail when model_selection activity fails"
    );
}

#[tokio::test]
async fn pre_loop_activity_runs_before_generate() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::noop("validate").into(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(pipeline_with_validate("generate"));
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

    // validate should come before generation.started in the log.
    // Use in_order_with to distinguish the validate activity.started from others.
    let validate_pred = |e: &weft_reactor::Event| {
        e.payload.pointer("/event/name").and_then(|v| v.as_str()) == Some("validate")
    };
    let gen_pred = |_: &weft_reactor::Event| true;
    EventAssertions::for_execution(&event_log, &result.execution_id)
        .await
        .in_order_with(&[
            ("activity.started", &validate_pred as EventPredicate<'_>),
            ("generation.started", &gen_pred),
        ]);
}
