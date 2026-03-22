//! Hook blocking, pre-response retry, and pre-generate hook integration tests.
//!
//! Tests covering hook blocking in pre-loop, pre-response hook block+retry,
//! pre-generate hook non-critical failure, critical failure, and blocked.

mod harness;

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor::activity::{Activity, ActivityInput};
use weft_reactor::config::{ActivityRef, LoopHooks, PipelineConfig};
use weft_reactor::error::ReactorError;
use weft_reactor::event::{
    ActivityEvent, FailureDetail, GeneratedEvent, GenerationEvent, HookOutcome, PipelineEvent,
};
use weft_reactor::event_log::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::{make_test_services, make_test_services_with_blocking_hook};
use weft_reactor::{ExecutionContext, RequestId, TenantId};

use weft_core::{HookEvent, Role, Source, WeftMessage};

use harness::{
    ImmediateDoneActivity, StubAssembleResponse, StubExecuteCommand, build_registry,
    reactor_config, test_event_log, test_request,
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

    struct HookStartActivity;

    #[async_trait::async_trait]
    impl Activity for HookStartActivity {
        fn name(&self) -> &str {
            "hook_request_start"
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name().to_string(),
                }))
                .await;

            let result = services
                .hooks()
                .run_chain(HookEvent::RequestStart, serde_json::json!({}), None)
                .await;

            match result {
                weft_hooks::HookChainResult::Blocked { hook_name, reason } => {
                    let _ = event_tx
                        .send(PipelineEvent::Hook(HookOutcome::Blocked {
                            hook_event: "request_start".to_string(),
                            hook_name,
                            reason,
                        }))
                        .await;
                }
                weft_hooks::HookChainResult::Allowed { .. } => {
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Completed {
                            name: self.name().to_string(),
                            idempotency_key: None,
                        }))
                        .await;
                }
            }
        }
    }

    let registry = build_registry(vec![
        Arc::new(HookStartActivity),
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
    let gen_calls_clone = Arc::clone(&gen_calls);
    let hook_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let hook_calls_clone = Arc::clone(&hook_calls);

    struct CountingDoneActivity {
        name: String,
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    #[async_trait::async_trait]
    impl Activity for CountingDoneActivity {
        fn name(&self) -> &str {
            &self.name
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
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name.clone(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                    GeneratedEvent::Done,
                )))
                .await;
            let response_message = WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("stub-model".to_string()),
                content: vec![],
                delta: false,
                message_index: 0,
            };
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Completed {
                    model: "stub-model".to_string(),
                    response_message,
                    generated_events: vec![GeneratedEvent::Done],
                    input_tokens: None,
                    output_tokens: None,
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: self.name.clone(),
                    idempotency_key: input.idempotency_key.clone(),
                }))
                .await;
        }
    }

    struct BlockOnceThenAllowHook {
        name: String,
        call_count: Arc<std::sync::atomic::AtomicU32>,
        block_reason: String,
    }

    #[async_trait::async_trait]
    impl Activity for BlockOnceThenAllowHook {
        fn name(&self) -> &str {
            &self.name
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
            let call_n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: self.name.clone(),
                }))
                .await;

            if call_n == 0 {
                let _ = event_tx
                    .send(PipelineEvent::Hook(HookOutcome::Blocked {
                        hook_event: "pre_response".to_string(),
                        hook_name: self.name.clone(),
                        reason: self.block_reason.clone(),
                    }))
                    .await;
            } else {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Completed {
                        name: self.name.clone(),
                        idempotency_key: input.idempotency_key.clone(),
                    }))
                    .await;
            }
        }
    }

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
        Arc::new(CountingDoneActivity {
            name: "generate".to_string(),
            call_count: gen_calls_clone,
        }),
        Arc::new(StubAssembleResponse {
            name: "assemble_response".to_string(),
        }),
        Arc::new(StubExecuteCommand {
            name: "execute_command".to_string(),
        }),
        Arc::new(BlockOnceThenAllowHook {
            name: "pre_response_hook".to_string(),
            call_count: hook_calls_clone,
            block_reason: "content policy violation: try again".to_string(),
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
    let events = event_log
        .read(&exec_result.execution_id, None::<u64>)
        .await
        .unwrap();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert!(
        event_types.contains(&"hook.blocked"),
        "event log should contain hook.blocked; got: {event_types:?}"
    );
}

#[tokio::test]
async fn pre_generate_hook_non_critical_failure_degrades_and_continues() {
    /// Stub hook activity: NonCritical, always emits ActivityEvent::Failed.
    struct NonCriticalFailingHookActivity;

    #[async_trait::async_trait]
    impl Activity for NonCriticalFailingHookActivity {
        fn name(&self) -> &str {
            "hook_pre_generate"
        }

        fn criticality(&self) -> weft_reactor_trait::Criticality {
            weft_reactor_trait::Criticality::NonCritical
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
                    name: "hook_pre_generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "hook_pre_generate".to_string(),
                    error: "hook runner unavailable".to_string(),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "hook_runner_error".to_string(),
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
        Arc::new(NonCriticalFailingHookActivity),
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

    // ExecutionEvent::Degraded must be recorded for the hook failure.
    let events = event_log
        .read(&exec_result.execution_id, None::<u64>)
        .await
        .unwrap_or_default();
    let degraded_events: Vec<_> = events
        .iter()
        .filter(|e| e.event_type == "execution.degraded")
        .collect();

    assert!(
        !degraded_events.is_empty(),
        "should have at least one execution.degraded event; event_types: {:?}",
        events
            .iter()
            .map(|e| e.event_type.as_str())
            .collect::<Vec<_>>()
    );

    // The degradation should reference hook_pre_generate.
    let has_hook_degradation = degraded_events.iter().any(|e| {
        e.payload
            .get("event")
            .and_then(|ev: &serde_json::Value| ev.get("notice"))
            .and_then(|n: &serde_json::Value| n.get("activity_name"))
            .and_then(|a: &serde_json::Value| a.as_str())
            == Some("hook_pre_generate")
    });
    assert!(
        has_hook_degradation,
        "degradation should be attributed to hook_pre_generate; degraded events: {degraded_events:?}"
    );
}

#[tokio::test]
async fn pre_generate_hook_critical_failure_terminates_execution() {
    /// Stub hook activity: Critical (default), always emits ActivityEvent::Failed.
    struct CriticalFailingHookActivity;

    #[async_trait::async_trait]
    impl Activity for CriticalFailingHookActivity {
        fn name(&self) -> &str {
            "hook_pre_generate"
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
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "hook_pre_generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: "hook_pre_generate".to_string(),
                    error: "critical hook failure".to_string(),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "hook_critical_error".to_string(),
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
        Arc::new(CriticalFailingHookActivity),
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
    /// Stub hook activity: NonCritical, emits HookOutcome::Blocked.
    ///
    /// Uses NonCritical to confirm that criticality is irrelevant for blocked events.
    struct BlockingNonCriticalHookActivity;

    #[async_trait::async_trait]
    impl Activity for BlockingNonCriticalHookActivity {
        fn name(&self) -> &str {
            "hook_pre_generate"
        }

        fn criticality(&self) -> weft_reactor_trait::Criticality {
            // Intentionally non-critical to verify that Blocked is always fatal.
            weft_reactor_trait::Criticality::NonCritical
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
                    name: "hook_pre_generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Hook(HookOutcome::Blocked {
                    hook_event: "pre_generate".to_string(),
                    hook_name: "policy_guard".to_string(),
                    reason: "blocked by pre_generate policy".to_string(),
                }))
                .await;
        }
    }

    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        Arc::new(BlockingNonCriticalHookActivity),
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
