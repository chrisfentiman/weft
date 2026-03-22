//! Pre-loop and post-loop activity orchestration.
//!
//! Runs the sequential pre-loop and post-loop activity passes within
//! `Reactor::execute`. Separated from `execute.rs` to keep each file focused.

use tracing::{Instrument, info_span, warn};

use crate::error::ReactorError;
use crate::event::{
    CommandFormat, DegradationNotice, ExecutionEvent, FailureDetail, PipelineEvent, PipelinePhase,
};
use crate::execution::ExecutionStatus;

use super::Reactor;
use super::drain::DrainOutcome;
use super::types::LoopContext;

impl Reactor {
    /// Run the pre-loop activities sequentially.
    ///
    /// Each activity is wrapped in an `activity` span that includes the
    /// activity name and phase ("pre_loop"). The entire pre-loop phase is
    /// wrapped in a `pre_loop` span for total phase timing.
    ///
    /// Returns `Err` if the execution should terminate due to a hard failure.
    /// Cancellation mid-loop is handled by checking `cancel.is_cancelled()`
    /// at the start of each iteration.
    pub(super) async fn run_pre_loop(
        &self,
        lctx: &mut LoopContext<'_>,
    ) -> Result<(), ReactorError> {
        let pre_loop_span = info_span!("pre_loop");

        async move {
            for resolved in &lctx.pipeline.pre_loop {
                if lctx.cancel.is_cancelled() {
                    break;
                }

                let mut input =
                    self.build_input(lctx.execution_id, lctx.state, lctx.request, resolved, None);
                // Per spec Section 6.3: inject per-activity metadata so downstream
                // activities receive the state accumulated by prior pre-loop activities.
                match resolved.activity.name() {
                    "provider_resolution" => {
                        input.metadata = serde_json::json!({
                            "selected_model": lctx.state.selected_model.as_deref().unwrap_or("")
                        });
                    }
                    "command_formatting" => {
                        let cmd_names: Vec<&str> = lctx
                            .state
                            .selected_commands
                            .iter()
                            .map(|c| c.name.as_str())
                            .collect();
                        input.metadata = serde_json::json!({
                            "capabilities": lctx.state.model_capabilities,
                            "selected_commands": cmd_names,
                        });
                    }
                    "sampling_adjustment" => {
                        input.metadata = serde_json::json!({
                            "max_tokens": lctx.state.model_max_tokens.unwrap_or(4096)
                        });
                    }
                    _ => {}
                }

                // Wrap each activity execution in an `activity` span. The reactor
                // is the orchestrator and knows the phase context ("pre_loop").
                // Activities themselves do not create this outer span.
                let activity_name = resolved.activity.name().to_string();
                let activity_span = info_span!(
                    "activity",
                    activity.name = %activity_name,
                    activity.phase = "pre_loop",
                    activity.status = tracing::field::Empty,
                );

                let activity_result = async {
                    resolved
                        .activity
                        .execute(
                            lctx.execution_id,
                            input,
                            self.services.as_ref(),
                            self.event_log.as_ref(),
                            lctx.event_tx.clone(),
                            lctx.cancel.clone(),
                        )
                        .await;

                    // Drain and dispatch all events the activity pushed.
                    self.drain_pre_post_loop(
                        lctx.execution_id,
                        lctx.event_rx,
                        lctx.state,
                        lctx.cancel,
                    )
                    .await
                }
                .instrument(activity_span.clone())
                .await;

                match activity_result {
                    Err(e) => {
                        activity_span.record("activity.status", "error");
                        return Err(e);
                    }
                    Ok(DrainOutcome::Continue) => {
                        activity_span.record("activity.status", "ok");
                    }
                    Ok(DrainOutcome::ActivityFailed { name, error, detail, .. }) => {
                        let criticality = resolved.activity.criticality();
                        if criticality.is_fatal() {
                            // Critical failure: terminate the request (unchanged behavior).
                            activity_span.record("activity.status", "error");
                            let err = ReactorError::ActivityFailed(
                                crate::activity::ActivityError::Failed {
                                    name: name.clone(),
                                    reason: error.clone(),
                                },
                            );
                            self.record_event(
                                lctx.execution_id,
                                &PipelineEvent::Execution(ExecutionEvent::Failed {
                                    error: err.to_string(),
                                }),
                                Some(
                                    &serde_json::json!({ "partial_text": lctx.state.accumulated_text }),
                                ),
                            )
                            .await?;
                            self.event_log
                                .update_execution_status(
                                    lctx.execution_id,
                                    ExecutionStatus::Failed,
                                )
                                .await?;
                            return Err(err);
                        } else {
                            // Non-critical or semi-critical: degrade and continue.
                            activity_span.record("activity.status", "degraded");
                            let notice = build_degradation_notice(
                                &name,
                                PipelinePhase::PreLoop,
                                &detail,
                                &error,
                                lctx.state.config.default_model.clone(),
                            );
                            // Apply default state for downstream activities.
                            apply_degradation_defaults(&name, lctx.state);
                            // Record Execution(Degraded) event.
                            self.record_event(
                                lctx.execution_id,
                                &PipelineEvent::Execution(ExecutionEvent::Degraded {
                                    notice: notice.clone(),
                                }),
                                None,
                            )
                            .await?;
                            // Emit warn span event with structured fields.
                            warn!(
                                degradation.activity = %notice.activity_name,
                                degradation.error_code = %notice.error_code,
                                degradation.phase = ?notice.phase,
                                degradation.fallback = %notice.fallback_applied,
                                "Activity degraded: {}", notice.message,
                            );
                            lctx.state.degradations.push(notice);
                        }
                    }
                    Ok(DrainOutcome::HookBlocked { hook_name, reason }) => {
                        // Hook-blocked is always fatal regardless of criticality.
                        activity_span.record("activity.status", "error");
                        let err = ReactorError::HookBlocked {
                            hook_name: hook_name.clone(),
                            reason: reason.clone(),
                        };
                        self.record_event(
                            lctx.execution_id,
                            &PipelineEvent::Execution(ExecutionEvent::Failed {
                                error: err.to_string(),
                            }),
                            Some(
                                &serde_json::json!({ "partial_text": lctx.state.accumulated_text }),
                            ),
                        )
                        .await?;
                        self.event_log
                            .update_execution_status(lctx.execution_id, ExecutionStatus::Failed)
                            .await?;
                        return Err(err);
                    }
                    Ok(DrainOutcome::Cancelled { reason }) => {
                        activity_span.record("activity.status", "error");
                        return Err(ReactorError::Cancelled { reason });
                    }
                }
            }
            Ok(())
        }
        .instrument(pre_loop_span)
        .await
    }

    /// Run the post-loop activities sequentially.
    ///
    /// Each activity is wrapped in an `activity` span that includes the
    /// activity name and phase ("post_loop"). The entire post-loop phase is
    /// wrapped in a `post_loop` span.
    ///
    /// Post-loop failures are logged as warnings but do not prevent
    /// returning partial results — execution has already succeeded at
    /// this point. Degradation notices are accumulated for observability parity.
    pub(super) async fn run_post_loop(
        &self,
        lctx: &mut LoopContext<'_>,
    ) -> Result<(), ReactorError> {
        let post_loop_span = info_span!("post_loop");

        async move {
            for resolved in &lctx.pipeline.post_loop {
                if lctx.cancel.is_cancelled() {
                    break;
                }
                let input =
                    self.build_input(lctx.execution_id, lctx.state, lctx.request, resolved, None);

                let activity_name = resolved.activity.name().to_string();
                let activity_span = info_span!(
                    "activity",
                    activity.name = %activity_name,
                    activity.phase = "post_loop",
                    activity.status = tracing::field::Empty,
                );

                let activity_result = async {
                    resolved
                        .activity
                        .execute(
                            lctx.execution_id,
                            input,
                            self.services.as_ref(),
                            self.event_log.as_ref(),
                            lctx.event_tx.clone(),
                            lctx.cancel.clone(),
                        )
                        .await;
                    self.drain_pre_post_loop(
                        lctx.execution_id,
                        lctx.event_rx,
                        lctx.state,
                        lctx.cancel,
                    )
                    .await
                }
                .instrument(activity_span.clone())
                .await;

                match activity_result {
                    Ok(DrainOutcome::Continue) => {
                        activity_span.record("activity.status", "ok");
                    }
                    Ok(DrainOutcome::ActivityFailed {
                        name,
                        error,
                        detail,
                        ..
                    }) => {
                        let criticality = resolved.activity.criticality();
                        if criticality.is_fatal() {
                            activity_span.record("activity.status", "error");
                            // Post-loop fatal failure: log warning and break (return partial results).
                            warn!("post-loop critical activity failed: {error}");
                            break;
                        } else {
                            // Non-critical: degrade, record notice, continue.
                            activity_span.record("activity.status", "degraded");
                            let notice = build_degradation_notice(
                                &name,
                                PipelinePhase::PostLoop,
                                &detail,
                                &error,
                                lctx.state.config.default_model.clone(),
                            );
                            apply_degradation_defaults(&name, lctx.state);
                            // Record the degradation event but swallow the error — we're
                            // already past the critical path.
                            let _ = self
                                .record_event(
                                    lctx.execution_id,
                                    &PipelineEvent::Execution(ExecutionEvent::Degraded {
                                        notice: notice.clone(),
                                    }),
                                    None,
                                )
                                .await;
                            warn!(
                                degradation.activity = %notice.activity_name,
                                degradation.error_code = %notice.error_code,
                                degradation.phase = ?notice.phase,
                                degradation.fallback = %notice.fallback_applied,
                                "Activity degraded: {}", notice.message,
                            );
                            lctx.state.degradations.push(notice);
                        }
                    }
                    Ok(DrainOutcome::HookBlocked { .. }) | Ok(DrainOutcome::Cancelled { .. }) => {
                        activity_span.record("activity.status", "error");
                        warn!("post-loop activity blocked or cancelled; stopping post-loop");
                        break;
                    }
                    Err(e) => {
                        activity_span.record("activity.status", "error");
                        warn!("post-loop drain error: {e}");
                        break;
                    }
                }
            }
            Ok(())
        }
        .instrument(post_loop_span)
        .await
    }
}

/// Build a `DegradationNotice` from a failed activity.
///
/// Uses the activity name to produce the appropriate human-readable message,
/// impact description, and fallback description per spec Section 4.4.
fn build_degradation_notice(
    activity_name: &str,
    phase: PipelinePhase,
    detail: &FailureDetail,
    _error: &str,
    default_model: String,
) -> DegradationNotice {
    let (message, impact, fallback_applied) = match activity_name {
        "model_selection" => (
            format!("Model selection failed; using default model '{default_model}'"),
            "Request uses default model instead of dynamically selected model".to_string(),
            format!("default model: {default_model}"),
        ),
        "command_selection" => (
            "Command selection failed; proceeding without commands".to_string(),
            "Commands will not be available for this request".to_string(),
            "empty command list".to_string(),
        ),
        "system_prompt_assembly" => (
            "System prompt assembly failed; proceeding without system prompt".to_string(),
            "LLM receives no system prompt context".to_string(),
            "no system prompt".to_string(),
        ),
        "command_formatting" => (
            "Command formatting failed; proceeding without command descriptions".to_string(),
            "Provider receives no command descriptions".to_string(),
            "no commands formatted".to_string(),
        ),
        "sampling_adjustment" => (
            "Sampling adjustment failed; using provider defaults".to_string(),
            "Sampling parameters use provider defaults".to_string(),
            "provider defaults".to_string(),
        ),
        // Non-critical hooks or any other non-critical activity.
        name => (
            format!("Hook '{name}' failed; skipping"),
            "Hook effect skipped".to_string(),
            "hook skipped".to_string(),
        ),
    };

    DegradationNotice {
        activity_name: activity_name.to_string(),
        phase,
        error_code: detail.error_code.clone(),
        message,
        impact,
        fallback_applied,
    }
}

/// Apply default state for downstream activities after a non-critical activity degrades.
///
/// Each entry mirrors spec Section 4.3. The activity declares *whether* it can
/// degrade; the reactor decides *what defaults to apply* when it does.
fn apply_degradation_defaults(activity_name: &str, state: &mut super::types::ExecutionState) {
    match activity_name {
        "model_selection" => {
            // Semi-critical: use default model with score 0.0 so ProviderResolutionActivity
            // can still look up a provider.
            let default_model = state.config.default_model.clone();
            state.selected_model = Some(default_model.clone());
            state.routing = Some(crate::activity::RoutingSnapshot {
                model_routing: weft_core::RoutingActivity {
                    model: default_model,
                    score: 0.0,
                    filters: vec![],
                },
                tool_necessity: None,
                tool_necessity_score: None,
            });
        }
        "command_selection" => {
            // Non-critical: empty command list so command_formatting sees no commands.
            state.selected_commands = vec![];
        }
        "system_prompt_assembly" => {
            // Non-critical: no state change — messages remain as-is.
            // The LLM receives the request without a system prompt.
        }
        "command_formatting" => {
            // Non-critical: set NoCommands so the provider sees no command descriptions.
            state.command_format = Some(CommandFormat::NoCommands);
        }
        "sampling_adjustment" => {
            // Non-critical: no state change — provider uses its defaults.
        }
        _ => {
            // Non-critical hook or other activity: no state change.
        }
    }
}
