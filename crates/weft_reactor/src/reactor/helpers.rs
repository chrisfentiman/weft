//! Helper methods on Reactor: input construction, event recording,
//! idempotency, and finalization.

use std::sync::Arc;

use crate::activity::ActivityInput;
use crate::error::ReactorError;
use crate::event::{ActivityEvent, EVENT_SCHEMA_VERSION, Event, ExecutionEvent, PipelineEvent};
use crate::execution::{ExecutionId, ExecutionStatus};

use super::Reactor;
use super::types::{ExecutionState, ResolvedActivity};

impl Reactor {
    /// Build an ActivityInput snapshot from current ExecutionState.
    ///
    /// The `execution_id` parameter is unused here but kept for future use
    /// (e.g., fetching per-execution config overrides).
    pub(super) fn build_input(
        &self,
        _execution_id: &ExecutionId,
        state: &ExecutionState,
        request: &weft_core::WeftRequest,
        _resolved: &ResolvedActivity,
        idempotency_key: Option<String>,
    ) -> ActivityInput {
        // Per spec Section 6.4: generation_config derives from override, then selected_model
        // (with sampling params), then falls back to routing snapshot for backward compat.
        let generation_config = state
            .generation_config_override
            .clone()
            .or_else(|| {
                state.selected_model.as_ref().map(|model| {
                    let mut config = serde_json::json!({ "model": model });
                    if let Some(max_tokens) = state.sampling_max_tokens {
                        config["max_tokens"] = serde_json::json!(max_tokens);
                    }
                    if let Some(temp) = state.sampling_temperature {
                        config["temperature"] = serde_json::json!(temp);
                    }
                    if let Some(top_p) = state.sampling_top_p {
                        config["top_p"] = serde_json::json!(top_p);
                    }
                    config
                })
            })
            .or_else(|| {
                // Backward compat: if selected_model not yet set (e.g., pre-loop not complete),
                // fall back to routing snapshot for serialized event log replay.
                state
                    .routing
                    .as_ref()
                    .map(|r| serde_json::json!({ "model": r.model_routing.model }))
            });

        // Per spec Section 6.4: metadata comes from ActivityRef.config().
        // ResolvedActivity carries the activity but not the original ActivityRef.
        // We use Value::Null as the default; callers that need specific metadata
        // (e.g., execute_commands) override the metadata field after calling build_input.
        let metadata = serde_json::Value::Null;

        // Build a ChildSpawner if the reactor handle is available (it's set after
        // construction). Activities that spawn child executions (e.g., GenerateActivity)
        // use this instead of accessing Services directly.
        let child_spawner: Option<Arc<dyn crate::activity::ChildSpawner>> =
            self.services.reactor_handle.get().map(|handle| {
                let spawner: Arc<dyn crate::activity::ChildSpawner> = Arc::new(
                    crate::services::ReactorChildSpawner::new(Arc::clone(handle)),
                );
                spawner
            });

        ActivityInput {
            messages: state.messages.clone(),
            request: request.clone(),
            routing_result: state.routing.clone(),
            budget: state.budget.clone(),
            metadata,
            generation_config,
            accumulated_text: state.accumulated_text.clone(),
            available_commands: state.available_commands.clone(),
            idempotency_key,
            accumulated_usage: state.accumulated_usage.clone(),
            child_spawner,
        }
    }

    /// Record a PipelineEvent to the EventLog.
    ///
    /// Derives event_type string from variant, serializes to JSON,
    /// and passes EVENT_SCHEMA_VERSION. Extracts idempotency_key from
    /// ActivityCompleted events if present.
    ///
    /// If `enrichment` is Some, its key-value pairs are merged into the
    /// `event` sub-object of the serialized payload with `_obs_` prefix.
    /// This preserves observability data that was removed from the typed
    /// variant (see spec Section 5.2). The `_obs_*` fields are silently
    /// dropped when deserializing stored JSON back to PipelineEvent.
    pub(super) async fn record_event(
        &self,
        execution_id: &ExecutionId,
        event: &PipelineEvent,
        enrichment: Option<&serde_json::Value>,
    ) -> Result<u64, ReactorError> {
        let event_type = event.event_type_string();
        let mut payload = serde_json::to_value(event)?;
        // Merge _obs_* enrichment fields into the event sub-object.
        if let (Some(enrichment), Some(obj)) = (enrichment, payload.as_object_mut())
            && let Some(event_obj) = obj.get_mut("event").and_then(|v| v.as_object_mut())
            && let Some(enrich_obj) = enrichment.as_object()
        {
            for (k, v) in enrich_obj {
                event_obj.insert(format!("_obs_{k}"), v.clone());
            }
        }
        let idempotency_key = match event {
            PipelineEvent::Activity(ActivityEvent::Completed {
                idempotency_key: Some(key),
                ..
            }) => Some(key.as_str()),
            _ => None,
        };
        let seq = self
            .event_log
            .append(
                execution_id,
                event_type,
                payload,
                EVENT_SCHEMA_VERSION,
                idempotency_key,
            )
            .await?;
        Ok(seq)
    }

    /// Check for a prior completion with the given idempotency key.
    ///
    /// Returns Some(events) if a prior completion exists (for replay),
    /// None if the activity should run fresh.
    pub(super) async fn check_idempotency(
        &self,
        execution_id: &ExecutionId,
        idempotency_key: &str,
    ) -> Result<Option<Vec<Event>>, ReactorError> {
        let events = self.event_log.read(execution_id, None).await?;

        // Find an ActivityCompleted event with this idempotency key.
        let matching = events.iter().find(|e| {
            e.event_type == "activity.completed"
                && e.payload.get("idempotency_key").and_then(|v| v.as_str())
                    == Some(idempotency_key)
        });

        if matching.is_some() {
            Ok(Some(self.extract_activity_events(idempotency_key, &events)))
        } else {
            Ok(None)
        }
    }

    /// Extract events produced by an activity between its ActivityStarted
    /// and ActivityCompleted markers, identified by idempotency_key.
    pub(super) fn extract_activity_events(
        &self,
        idempotency_key: &str,
        events: &[Event],
    ) -> Vec<Event> {
        // Find the ActivityCompleted event with this key.
        let completed_seq = events
            .iter()
            .find(|e| {
                e.event_type == "activity.completed"
                    && e.payload.get("idempotency_key").and_then(|v| v.as_str())
                        == Some(idempotency_key)
            })
            .map(|e| e.sequence);

        let Some(completed_seq) = completed_seq else {
            return Vec::new();
        };

        // Find the ActivityStarted event just before the ActivityCompleted.
        // We look for the last ActivityStarted at sequence < completed_seq.
        let started_seq = events
            .iter()
            .filter(|e| e.event_type == "activity.started" && e.sequence < completed_seq)
            .map(|e| e.sequence)
            .max();

        let Some(started_seq) = started_seq else {
            return Vec::new();
        };

        // Return all events between started_seq and completed_seq (inclusive).
        events
            .iter()
            .filter(|e| e.sequence >= started_seq && e.sequence <= completed_seq)
            .cloned()
            .collect()
    }

    /// Record ExecutionFailed and update status. Used as a helper in error paths.
    pub(super) async fn finalize_failed(
        &self,
        execution_id: &ExecutionId,
        state: &mut ExecutionState,
        err: ReactorError,
    ) -> Result<(), ReactorError> {
        self.record_event(
            execution_id,
            &PipelineEvent::Execution(ExecutionEvent::Failed {
                error: err.to_string(),
            }),
            Some(&serde_json::json!({ "partial_text": state.accumulated_text })),
        )
        .await?;
        self.event_log
            .update_execution_status(execution_id, ExecutionStatus::Failed)
            .await?;
        Ok(())
    }
}

/// Build an empty WeftResponse for cancelled/failed executions with no assembled response.
pub(super) fn empty_response(execution_id: &ExecutionId) -> weft_core::WeftResponse {
    weft_core::WeftResponse {
        id: execution_id.to_string(),
        model: "unknown".to_string(),
        messages: vec![],
        usage: weft_core::WeftUsage::default(),
        timing: weft_core::WeftTiming::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::activity::{Activity, ActivityInput};
    use crate::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
    use crate::event::{ActivityEvent, PipelineEvent};
    use crate::event_log::EventLog;
    use crate::execution::{Execution, ExecutionId, ExecutionStatus};
    use crate::registry::ActivityRegistry;
    use crate::test_support::{TestEventLog, make_test_services};

    use super::super::Reactor;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_core::{ModelRoutingInstruction, SamplingOptions, WeftRequest};

    // ── Minimal stubs ─────────────────────────────────────────────────────

    struct MinimalDoneActivity;

    #[async_trait::async_trait]
    impl Activity for MinimalDoneActivity {
        fn name(&self) -> &str {
            "generate"
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
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "generate".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: "generate".to_string(),
                    idempotency_key: input.idempotency_key.clone(),
                }))
                .await;
        }
    }

    struct MinimalAssembleResponse;

    #[async_trait::async_trait]
    impl Activity for MinimalAssembleResponse {
        fn name(&self) -> &str {
            "assemble_response"
        }

        async fn execute(
            &self,
            execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn weft_reactor_trait::ServiceLocator,
            _event_log: &dyn EventLog,
            event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Started {
                    name: "assemble_response".to_string(),
                }))
                .await;
            let response = weft_core::WeftResponse {
                id: execution_id.to_string(),
                model: "stub".to_string(),
                messages: vec![],
                usage: weft_core::WeftUsage::default(),
                timing: weft_core::WeftTiming::default(),
            };
            let _ = event_tx
                .send(PipelineEvent::Context(
                    crate::event::ContextEvent::ResponseAssembled { response },
                ))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: "assemble_response".to_string(),
                    idempotency_key: None,
                }))
                .await;
        }
    }

    struct MinimalExecuteCommand;

    #[async_trait::async_trait]
    impl Activity for MinimalExecuteCommand {
        fn name(&self) -> &str {
            "execute_command"
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
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: "execute_command".to_string(),
                    idempotency_key: input.idempotency_key.clone(),
                }))
                .await;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn simple_pipeline() -> PipelineConfig {
        PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::Name("generate".to_string()),
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }
    }

    fn test_reactor_config() -> ReactorConfig {
        ReactorConfig {
            pipelines: vec![simple_pipeline()],
            budget: BudgetConfig {
                max_generation_calls: 5,
                max_iterations: 5,
                max_depth: 3,
                timeout_secs: 300,
                generation_timeout_secs: 60,
                command_timeout_secs: 10,
            },
        }
    }

    fn build_test_reactor(event_log: Arc<TestEventLog>) -> Reactor {
        let services = Arc::new(make_test_services());
        let mut registry = ActivityRegistry::new();
        registry
            .register(Arc::new(MinimalDoneActivity))
            .expect("duplicate in test registry");
        registry
            .register(Arc::new(MinimalAssembleResponse))
            .expect("duplicate in test registry");
        registry
            .register(Arc::new(MinimalExecuteCommand))
            .expect("duplicate in test registry");
        let registry = Arc::new(registry);
        let config = test_reactor_config();
        Reactor::new(services, event_log, registry, &config).expect("reactor build should succeed")
    }

    fn make_event_log() -> Arc<TestEventLog> {
        Arc::new(TestEventLog::new())
    }

    // ── Tests: check_idempotency ──────────────────────────────────────────

    /// check_idempotency returns Some(events) when the key exists in the log.
    /// The returned slice must include both activity.started and activity.completed.
    #[tokio::test]
    async fn check_idempotency_finds_completed_activity() {
        let event_log = make_event_log();
        let reactor = build_test_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        let key = format!("{}:generate:0", execution_id);

        event_log
            .create_execution(&Execution {
                id: execution_id.clone(),
                tenant_id: crate::execution::TenantId("t1".to_string()),
                request_id: crate::execution::RequestId("r1".to_string()),
                parent_id: None,
                pipeline_name: "default".to_string(),
                status: ExecutionStatus::Running,
                created_at: chrono::Utc::now(),
                depth: 0,
            })
            .await
            .unwrap();

        event_log
            .append(
                &execution_id,
                "activity.started",
                serde_json::json!({ "name": "generate" }),
                1,
                None,
            )
            .await
            .unwrap();

        event_log
            .append(
                &execution_id,
                "activity.completed",
                serde_json::json!({
                    "name": "generate",
                    "duration_ms": 1,
                    "idempotency_key": key
                }),
                1,
                Some(key.as_str()),
            )
            .await
            .unwrap();

        let result = reactor
            .check_idempotency(&execution_id, &key)
            .await
            .expect("check_idempotency should not fail");

        assert!(
            result.is_some(),
            "should return Some when idempotency key is present"
        );
        let events = result.unwrap();
        assert!(!events.is_empty(), "replayed events should not be empty");
        let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
        assert!(
            types.contains(&"activity.started"),
            "replayed events should include activity.started; got: {types:?}"
        );
        assert!(
            types.contains(&"activity.completed"),
            "replayed events should include activity.completed; got: {types:?}"
        );
    }

    /// check_idempotency returns None when the key is absent from the log.
    #[tokio::test]
    async fn check_idempotency_miss_returns_none() {
        let event_log = make_event_log();
        let reactor = build_test_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();

        event_log
            .create_execution(&Execution {
                id: execution_id.clone(),
                tenant_id: crate::execution::TenantId("t1".to_string()),
                request_id: crate::execution::RequestId("r1".to_string()),
                parent_id: None,
                pipeline_name: "default".to_string(),
                status: ExecutionStatus::Running,
                created_at: chrono::Utc::now(),
                depth: 0,
            })
            .await
            .unwrap();

        let missing_key = format!("{}:generate:99", execution_id);
        let result = reactor
            .check_idempotency(&execution_id, &missing_key)
            .await
            .expect("check_idempotency should not fail on miss");

        assert!(result.is_none(), "should return None when key is absent");
    }

    // ── Tests: build_input generation_config cascade ──────────────────────

    /// build_input uses generation_config_override when set (highest priority).
    #[test]
    fn build_input_uses_override_when_set() {
        use super::super::types::ExecutionState;
        use crate::budget::Budget;

        let event_log = make_event_log();
        let reactor = build_test_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();

        let mut state = ExecutionState::new(Budget::new(
            5,
            5,
            3,
            chrono::Utc::now() + chrono::Duration::hours(1),
        ));
        let override_config = serde_json::json!({ "model": "override-model", "temperature": 0.5 });
        state.generation_config_override = Some(override_config.clone());
        state.selected_model = Some("selected-model".to_string());

        let request = WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };

        // We need a ResolvedActivity. Since it's pub(super), access via a helper
        // that constructs one from the registry. Instead, test indirectly through
        // the ActivityInput returned.
        let registry = {
            let mut r = ActivityRegistry::new();
            r.register(Arc::new(MinimalDoneActivity)).unwrap();
            r.register(Arc::new(MinimalAssembleResponse)).unwrap();
            r.register(Arc::new(MinimalExecuteCommand)).unwrap();
            Arc::new(r)
        };
        let config = test_reactor_config();
        // Use compile_pipeline to get a ResolvedActivity for "generate".
        let compiled = Reactor::compile_pipeline(&config.pipelines[0], &registry).expect("compile");

        let input = reactor.build_input(&execution_id, &state, &request, &compiled.generate, None);

        assert_eq!(
            input.generation_config.as_ref().unwrap()["model"],
            "override-model",
            "override should take priority over selected_model"
        );
    }

    /// build_input falls back to selected_model when no override is set.
    #[test]
    fn build_input_falls_back_to_selected_model() {
        use super::super::types::ExecutionState;
        use crate::budget::Budget;

        let event_log = make_event_log();
        let reactor = build_test_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();

        let mut state = ExecutionState::new(Budget::new(
            5,
            5,
            3,
            chrono::Utc::now() + chrono::Duration::hours(1),
        ));
        state.generation_config_override = None;
        state.selected_model = Some("claude-3".to_string());

        let request = WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };

        let registry = {
            let mut r = ActivityRegistry::new();
            r.register(Arc::new(MinimalDoneActivity)).unwrap();
            r.register(Arc::new(MinimalAssembleResponse)).unwrap();
            r.register(Arc::new(MinimalExecuteCommand)).unwrap();
            Arc::new(r)
        };
        let config = test_reactor_config();
        let compiled = Reactor::compile_pipeline(&config.pipelines[0], &registry).expect("compile");

        let input = reactor.build_input(&execution_id, &state, &request, &compiled.generate, None);

        assert_eq!(
            input.generation_config.as_ref().unwrap()["model"],
            "claude-3",
            "should fall back to selected_model when no override"
        );
    }

    /// build_input falls back to routing snapshot when neither override nor selected_model is set.
    #[test]
    fn build_input_falls_back_to_routing_snapshot() {
        use super::super::types::ExecutionState;
        use crate::activity::RoutingSnapshot;
        use crate::budget::Budget;

        let event_log = make_event_log();
        let reactor = build_test_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();

        let mut state = ExecutionState::new(Budget::new(
            5,
            5,
            3,
            chrono::Utc::now() + chrono::Duration::hours(1),
        ));
        state.generation_config_override = None;
        state.selected_model = None;
        state.routing = Some(RoutingSnapshot {
            model_routing: weft_core::RoutingActivity {
                model: "routing-fallback-model".to_string(),
                score: 0.9,
                filters: vec![],
            },
            tool_necessity: None,
            tool_necessity_score: None,
        });

        let request = WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };

        let registry = {
            let mut r = ActivityRegistry::new();
            r.register(Arc::new(MinimalDoneActivity)).unwrap();
            r.register(Arc::new(MinimalAssembleResponse)).unwrap();
            r.register(Arc::new(MinimalExecuteCommand)).unwrap();
            Arc::new(r)
        };
        let config = test_reactor_config();
        let compiled = Reactor::compile_pipeline(&config.pipelines[0], &registry).expect("compile");

        let input = reactor.build_input(&execution_id, &state, &request, &compiled.generate, None);

        assert_eq!(
            input.generation_config.as_ref().unwrap()["model"],
            "routing-fallback-model",
            "should fall back to routing snapshot when override and selected_model are absent"
        );
    }

    /// build_input returns None generation_config when all sources are absent.
    #[test]
    fn build_input_none_generation_config_when_all_absent() {
        use super::super::types::ExecutionState;
        use crate::budget::Budget;

        let event_log = make_event_log();
        let reactor = build_test_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();

        let state = ExecutionState::new(Budget::new(
            5,
            5,
            3,
            chrono::Utc::now() + chrono::Duration::hours(1),
        ));
        // All sources absent: override=None, selected_model=None, routing=None

        let request = WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };

        let registry = {
            let mut r = ActivityRegistry::new();
            r.register(Arc::new(MinimalDoneActivity)).unwrap();
            r.register(Arc::new(MinimalAssembleResponse)).unwrap();
            r.register(Arc::new(MinimalExecuteCommand)).unwrap();
            Arc::new(r)
        };
        let config = test_reactor_config();
        let compiled = Reactor::compile_pipeline(&config.pipelines[0], &registry).expect("compile");

        let input = reactor.build_input(&execution_id, &state, &request, &compiled.generate, None);

        assert!(
            input.generation_config.is_none(),
            "generation_config should be None when no source provides a model"
        );
    }
}
