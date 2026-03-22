//! Degradation paths and ExecutionResult population integration tests.
//!
//! Tests covering non-critical activity degradation, critical activity failure,
//! multiple degradation accumulation, semi-critical fallback, and
//! ExecutionResult.degradations propagation.

mod harness;

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::config::{ActivityRef, LoopHooks, PipelineConfig};
use weft_reactor::event::{ActivityEvent, FailureDetail, PipelineEvent, SelectionEvent};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use harness::{
    FailingNonCriticalActivity, FailingSemiCriticalActivity, ImmediateDoneActivity,
    StubAssembleResponse, StubCommandFormattingActivity, StubCommandSelectionActivity,
    StubExecuteCommand, StubModelSelectionActivity, StubProviderResolutionActivity,
    StubSamplingAdjustmentActivity, StubSystemPromptAssemblyActivity, StubValidateActivity,
    build_registry, new_preloop_pipeline_config, reactor_config, test_event_log, test_request,
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
        Arc::new(StubValidateActivity),
        Arc::new(StubModelSelectionActivity),
        // command_selection fails with NonCritical → should degrade and continue
        Arc::new(FailingNonCriticalActivity {
            activity_name: "command_selection".to_string(),
            error_code: "classifier_unavailable".to_string(),
        }),
        Arc::new(StubProviderResolutionActivity),
        Arc::new(StubSystemPromptAssemblyActivity),
        Arc::new(StubCommandFormattingActivity),
        Arc::new(StubSamplingAdjustmentActivity),
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
    // Payload structure: {"category": "Execution", "event": {"type": "Degraded", "notice": {...}}}
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
    struct FailingCriticalActivity;

    #[async_trait::async_trait]
    impl Activity for FailingCriticalActivity {
        fn name(&self) -> &str {
            "validate"
        }
        // criticality() defaults to Critical — no override needed.
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
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "validate".to_string(),
                    error: "empty messages".to_string(),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "empty_messages".to_string(),
                        detail: serde_json::Value::Null,
                        cause: None,
                        attempted: None,
                        fallback: None,
                    },
                }))
                .await;
        }
    }

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(FailingCriticalActivity),
        Arc::new(StubModelSelectionActivity),
        Arc::new(StubCommandSelectionActivity),
        Arc::new(StubProviderResolutionActivity),
        Arc::new(StubSystemPromptAssemblyActivity),
        Arc::new(StubCommandFormattingActivity),
        Arc::new(StubSamplingAdjustmentActivity),
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
        Arc::new(StubValidateActivity),
        Arc::new(StubModelSelectionActivity),
        // Two non-critical failures.
        Arc::new(FailingNonCriticalActivity {
            activity_name: "command_selection".to_string(),
            error_code: "classifier_unavailable".to_string(),
        }),
        Arc::new(StubProviderResolutionActivity),
        Arc::new(FailingNonCriticalActivity {
            activity_name: "system_prompt_assembly".to_string(),
            error_code: "template_error".to_string(),
        }),
        Arc::new(StubCommandFormattingActivity),
        Arc::new(StubSamplingAdjustmentActivity),
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
    // This ProviderResolutionActivity captures the model_selection result from
    // metadata injected by the reactor.
    struct CapturingProviderResolution {
        received_model: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    }

    #[async_trait::async_trait]
    impl Activity for CapturingProviderResolution {
        fn name(&self) -> &str {
            "provider_resolution"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            input: ActivityInput,
            _services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            // Capture what selected_model was injected as metadata.
            let selected = input
                .metadata
                .get("selected_model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            *self.received_model.lock().unwrap() = selected;

            // Emit ProviderResolved with the received model to allow execution to continue.
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name().to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                    model_name: "stub-model".to_string(),
                    model_id: "stub-model-v1".to_string(),
                    provider_name: "stub".to_string(),
                    capabilities: vec![],
                    max_tokens: 4096,
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: self.name().to_string(),
                    idempotency_key: None,
                }))
                .await;
        }
    }

    let received_model = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
    let received_model_clone = std::sync::Arc::clone(&received_model);

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(StubValidateActivity),
        // model_selection fails with SemiCritical → should use default model
        Arc::new(FailingSemiCriticalActivity {
            activity_name: "model_selection".to_string(),
        }),
        Arc::new(StubCommandSelectionActivity),
        Arc::new(CapturingProviderResolution {
            received_model: received_model_clone,
        }),
        Arc::new(StubSystemPromptAssemblyActivity),
        Arc::new(StubCommandFormattingActivity),
        Arc::new(StubSamplingAdjustmentActivity),
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
        Arc::new(FailingNonCriticalActivity {
            activity_name: "command_selection".to_string(),
            error_code: "classifier_unavailable".to_string(),
        }),
        Arc::new(StubModelSelectionActivity),
        Arc::new(StubProviderResolutionActivity),
        Arc::new(StubCommandFormattingActivity),
        Arc::new(StubSamplingAdjustmentActivity),
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
