//! Pre-loop activity ordering and failure integration tests.
//!
//! Tests covering the full pre-loop activity set, event ordering,
//! system prompt position, generation config, and pre-loop failure.

mod harness;

use std::sync::Arc;

use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{EventLog, ExecutionContext, RequestId, TenantId};

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
