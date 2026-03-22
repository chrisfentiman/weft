//! Degradation paths and ExecutionResult population integration tests.
//!
//! Tests covering non-critical activity degradation, critical activity failure,
//! multiple degradation accumulation, semi-critical fallback, and
//! ExecutionResult.degradations propagation.

mod harness;

use std::sync::{Arc, Mutex};

use weft_reactor::config::{ActivityRef, LoopHooks, PipelineConfig};
use weft_reactor::event::{FailureDetail, PipelineEvent, SelectionEvent};
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{EventLog, ExecutionContext, RequestId, TenantId};
use weft_reactor_trait::Criticality;

use harness::{
    TestActivity, build_registry, new_preloop_pipeline_config, reactor_config, test_event_log,
    test_request,
};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

/// Non-critical command_selection failure: execution completes with degradation.
///
/// Verifies spec Section 4.2: non-critical activity degrades, records
/// ExecutionEvent::Degraded, and execution continues.
#[tokio::test]
async fn non_critical_command_selection_failure_degrades_and_continues() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::validate_stub(),
        TestActivity::model_selection_stub(),
        // command_selection fails with NonCritical → should degrade and continue
        TestActivity::failing("command_selection")
            .with_criticality(Criticality::NonCritical)
            .with_detail(FailureDetail {
                error_code: "classifier_unavailable".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::provider_resolution_stub(),
        TestActivity::system_prompt_assembly_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(new_preloop_pipeline_config("generate"));
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

    // Non-critical failure should NOT terminate execution.
    let result =
        result.expect("non-critical command_selection failure should not terminate execution");

    // Verify ExecutionEvent::Degraded was recorded.
    let events = event_log
        .read(&result.0.execution_id, None::<u64>)
        .await
        .unwrap_or_default();
    let degraded_events: Vec<_> = events
        .iter()
        .filter(|e| e.event_type == "execution.degraded")
        .collect();
    assert!(
        !degraded_events.is_empty(),
        "should have at least one execution.degraded event"
    );

    // The degradation should mention command_selection.
    let has_command_selection_degradation = degraded_events.iter().any(|e| {
        e.payload
            .get("event")
            .and_then(|ev: &serde_json::Value| ev.get("notice"))
            .and_then(|n: &serde_json::Value| n.get("activity_name"))
            .and_then(|a: &serde_json::Value| a.as_str())
            == Some("command_selection")
    });
    assert!(
        has_command_selection_degradation,
        "degradation should be for command_selection; events: {degraded_events:?}"
    );
}

/// Critical activity failure terminates the request.
///
/// Verifies spec Section 4.2: Critical activities still fail the request.
#[tokio::test]
async fn critical_activity_failure_terminates_execution() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        // FailingCriticalActivity: Critical (default), emits Failed.
        TestActivity::failing("validate")
            .with_error("empty messages")
            .with_detail(FailureDetail {
                error_code: "empty_messages".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::model_selection_stub(),
        TestActivity::command_selection_stub(),
        TestActivity::provider_resolution_stub(),
        TestActivity::system_prompt_assembly_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(new_preloop_pipeline_config("generate"));
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
        "critical validate failure should terminate execution"
    );
}

/// Multiple non-critical degradations accumulate.
///
/// Verifies spec Section 7.3: multiple degradations are recorded separately.
#[tokio::test]
async fn multiple_non_critical_degradations_accumulate() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::validate_stub(),
        TestActivity::model_selection_stub(),
        // Two non-critical failures.
        TestActivity::failing("command_selection")
            .with_criticality(Criticality::NonCritical)
            .with_detail(FailureDetail {
                error_code: "classifier_unavailable".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::provider_resolution_stub(),
        TestActivity::failing("system_prompt_assembly")
            .with_criticality(Criticality::NonCritical)
            .with_detail(FailureDetail {
                error_code: "template_error".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(new_preloop_pipeline_config("generate"));
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

    let result = result.expect("multiple non-critical failures should not terminate execution");

    let events = event_log
        .read(&result.0.execution_id, None::<u64>)
        .await
        .unwrap_or_default();
    let degraded_events: Vec<_> = events
        .iter()
        .filter(|e| e.event_type == "execution.degraded")
        .collect();

    assert_eq!(
        degraded_events.len(),
        2,
        "should have 2 degradation events, one per failed non-critical activity; got: {degraded_events:?}"
    );
}

/// Semi-critical model_selection failure uses the default model fallback.
///
/// Verifies spec Section 4.3: model_selection degradation sets selected_model
/// to config.default_model. We verify this indirectly by confirming execution
/// succeeds (ProviderResolutionActivity receives a non-empty selected_model).
#[tokio::test]
async fn semi_critical_model_selection_uses_default_model() {
    let received_model = Arc::new(Mutex::new(None::<String>));
    let received_model_clone = Arc::clone(&received_model);

    // CaptureAndEmit: captures selected_model from metadata, emits ProviderResolved.
    let provider_resolution = Arc::new(harness::TestActivity {
        activity_name: "provider_resolution".to_string(),
        criticality: Criticality::Critical,
        behavior: harness::Behavior::CaptureAndEmit {
            capture_field: "selected_model".to_string(),
            captured: received_model_clone,
            events: vec![PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                model_name: "stub-model".to_string(),
                model_id: "stub-model-v1".to_string(),
                provider_name: "stub".to_string(),
                capabilities: vec![],
                max_tokens: 4096,
            })],
        },
    });

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::validate_stub(),
        // model_selection fails with SemiCritical → should use default model
        TestActivity::failing("model_selection")
            .with_criticality(Criticality::SemiCritical)
            .with_detail(FailureDetail {
                error_code: "no_matching_model".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::command_selection_stub(),
        provider_resolution,
        TestActivity::system_prompt_assembly_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(new_preloop_pipeline_config("generate"));
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
        result.is_ok(),
        "semi-critical model_selection failure should degrade and continue"
    );

    // Provider resolution should have received a non-empty model name (the default model).
    let captured = received_model.lock().unwrap().clone();
    assert!(
        captured.is_some() && !captured.as_deref().unwrap_or("").is_empty(),
        "provider_resolution should receive a non-empty model from default fallback; got: {captured:?}"
    );
}

/// ExecutionResult.degradations is empty on a fully successful execution.
///
/// Verifies that the degradations list is not spuriously populated when all
/// activities succeed.
#[tokio::test]
async fn execution_result_degradations_empty_on_success() {
    use pretty_assertions::assert_eq;

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::model_selection_stub(),
        TestActivity::command_selection_stub(),
        TestActivity::provider_resolution_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![
            ActivityRef::Name("model_selection".to_string()),
            ActivityRef::Name("command_selection".to_string()),
            ActivityRef::Name("provider_resolution".to_string()),
            ActivityRef::Name("command_formatting".to_string()),
            ActivityRef::Name("sampling_adjustment".to_string()),
        ],
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
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id: None,
                parent_budget: None,
                client_tx: None,
            },
            None,
        )
        .await;

    let (execution_result, _) = result.expect("successful execution should return Ok");
    assert_eq!(
        execution_result.degradations.len(),
        0,
        "no degradations on a fully successful execution"
    );
}

/// ExecutionResult.degradations is populated when a non-critical activity fails.
///
/// Verifies Phase 3 spec: degradations accumulated in ExecutionState are moved
/// to ExecutionResult via std::mem::take.
#[tokio::test]
async fn execution_result_degradations_populated_from_state() {
    use pretty_assertions::assert_eq;

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        // command_selection fails non-critically → should add a DegradationNotice
        TestActivity::failing("command_selection")
            .with_criticality(Criticality::NonCritical)
            .with_detail(FailureDetail {
                error_code: "classifier_unavailable".to_string(),
                detail: serde_json::Value::Null,
                cause: None,
                attempted: None,
                fallback: None,
            })
            .build(),
        TestActivity::model_selection_stub(),
        TestActivity::provider_resolution_stub(),
        TestActivity::command_formatting_stub(),
        TestActivity::sampling_adjustment_stub(),
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);

    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![
            ActivityRef::Name("command_selection".to_string()),
            ActivityRef::Name("model_selection".to_string()),
            ActivityRef::Name("provider_resolution".to_string()),
            ActivityRef::Name("command_formatting".to_string()),
            ActivityRef::Name("sampling_adjustment".to_string()),
        ],
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
                tenant_id: TenantId("t1".to_string()),
                request_id: RequestId("r1".to_string()),
                parent_id: None,
                parent_budget: None,
                client_tx: None,
            },
            None,
        )
        .await;

    let (execution_result, _) =
        result.expect("non-critical failure should not terminate execution");
    assert_eq!(
        execution_result.degradations.len(),
        1,
        "should have exactly one DegradationNotice for the failed command_selection"
    );
    assert_eq!(
        execution_result.degradations[0].activity_name, "command_selection",
        "degradation should be for command_selection"
    );
    assert_eq!(
        execution_result.degradations[0].error_code, "classifier_unavailable",
        "error_code should be propagated from FailureDetail"
    );
}
