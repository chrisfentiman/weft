//! Pre-loop and post-loop activity orchestration.
//!
//! Runs the sequential pre-loop and post-loop activity passes within
//! `Reactor::execute`. Separated from `execute.rs` to keep each file focused.

use tracing::warn;

use crate::error::ReactorError;
use crate::event::{ExecutionEvent, PipelineEvent};
use crate::execution::ExecutionStatus;

use super::Reactor;
use super::types::LoopContext;

impl Reactor {
    /// Run the pre-loop activities sequentially.
    ///
    /// Returns `Err` if the execution should terminate due to a hard failure.
    /// Cancellation mid-loop is handled by checking `cancel.is_cancelled()`
    /// at the start of each iteration.
    pub(super) async fn run_pre_loop(
        &self,
        lctx: &mut LoopContext<'_>,
    ) -> Result<(), ReactorError> {
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
            let terminate = self
                .drain_pre_post_loop(lctx.execution_id, lctx.event_rx, lctx.state, lctx.cancel)
                .await?;
            if let Some(err) = terminate {
                self.record_event(
                    lctx.execution_id,
                    &PipelineEvent::Execution(ExecutionEvent::Failed {
                        error: err.to_string(),
                    }),
                    Some(&serde_json::json!({ "partial_text": lctx.state.accumulated_text })),
                )
                .await?;
                self.event_log
                    .update_execution_status(lctx.execution_id, ExecutionStatus::Failed)
                    .await?;
                return Err(err);
            }
        }
        Ok(())
    }

    /// Run the post-loop activities sequentially.
    ///
    /// Post-loop failures are logged as warnings but do not prevent
    /// returning partial results — execution has already succeeded at
    /// this point.
    pub(super) async fn run_post_loop(
        &self,
        lctx: &mut LoopContext<'_>,
    ) -> Result<(), ReactorError> {
        for resolved in &lctx.pipeline.post_loop {
            if lctx.cancel.is_cancelled() {
                break;
            }
            let input =
                self.build_input(lctx.execution_id, lctx.state, lctx.request, resolved, None);
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
            let terminate = self
                .drain_pre_post_loop(lctx.execution_id, lctx.event_rx, lctx.state, lctx.cancel)
                .await?;
            if let Some(err) = terminate {
                // Post-loop failure: record but still return partial results.
                warn!("post-loop activity failed: {err}");
                break;
            }
        }
        Ok(())
    }
}
