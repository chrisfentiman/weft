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
