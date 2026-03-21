//! Pre-loop and post-loop activity orchestration.
//!
//! Runs the sequential pre-loop and post-loop activity passes within
//! `Reactor::execute`. Separated from `execute.rs` to keep each file focused.

use tracing::{Instrument, info_span, warn};

use crate::error::ReactorError;
use crate::event::{ExecutionEvent, PipelineEvent};
use crate::execution::ExecutionStatus;

use super::Reactor;
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
                    Ok(None) => {
                        activity_span.record("activity.status", "ok");
                    }
                    Ok(Some(err)) => {
                        activity_span.record("activity.status", "error");
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
                    Err(e) => {
                        activity_span.record("activity.status", "error");
                        return Err(e);
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
    /// this point.
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
                    Ok(None) => {
                        activity_span.record("activity.status", "ok");
                    }
                    Ok(Some(err)) => {
                        activity_span.record("activity.status", "error");
                        // Post-loop failure: record but still return partial results.
                        warn!("post-loop activity failed: {err}");
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
