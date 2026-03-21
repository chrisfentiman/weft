//! The `'dispatch` outer loop.
//!
//! Orchestrates each iteration of the dispatch loop: budget checks,
//! pre_generate hooks, idempotency replay, generate, retry, and commands.
//! The `'generate` inner loop lives in `generate.rs` and is invoked via
//! `run_generate_loop`, which returns a `GenerateOutcome` enum.

use std::time::Duration;

use tokio::sync::mpsc;
use tracing::debug;

use crate::budget::BudgetCheck;
use crate::error::ReactorError;
use crate::event::{
    ActivityEvent, BudgetEvent, ContextEvent, ExecutionEvent, GeneratedEvent, GenerationEvent,
    MessageInjectionSource, PipelineEvent,
};
use crate::execution::ExecutionStatus;

use super::generate::GenerateOutcome;
use super::helpers::empty_response;
use super::retry::{backoff_ms, should_retry};
use super::types::{BudgetUsage, CommandContext, ExecutionResult, LoopContext};
use super::Reactor;

impl Reactor {
    /// Run the dispatch loop (Phase 2 of execution).
    ///
    /// Orchestrates budget checks, pre_generate hooks, idempotency replay,
    /// generation (via `run_generate_loop`), retry logic, and command execution.
    ///
    /// Returns `Ok(Some(result))` on early exit (cancellation), `Ok(None)` on
    /// normal completion (budget exhaustion / Done / Refused), or `Err` on failure.
    pub(super) async fn run_dispatch_loop(
        &self,
        lctx: &mut LoopContext<'_>,
        client_tx: &Option<mpsc::Sender<PipelineEvent>>,
        default_generation_timeout: u64,
        default_command_timeout: u64,
    ) -> Result<Option<ExecutionResult>, ReactorError> {
        'dispatch: loop {
            // 2a. Budget check.
            match lctx.state.budget.check() {
                BudgetCheck::Exhausted(reason) => {
                    let resource = reason.to_string();
                    self.record_event(
                        lctx.execution_id,
                        &PipelineEvent::Budget(BudgetEvent::Exhausted {
                            resource: resource.clone(),
                        }),
                        None,
                    )
                    .await?;
                    debug!("budget exhausted: {resource}");
                    break 'dispatch;
                }
                BudgetCheck::Warning(info) => {
                    self.record_event(
                        lctx.execution_id,
                        &PipelineEvent::Budget(BudgetEvent::Warning {
                            resource: info.resource.clone(),
                            remaining: info.remaining,
                        }),
                        None,
                    )
                    .await?;
                }
                BudgetCheck::Ok => {}
            }

            // 2b. Run pre_generate hooks.
            for hook in &lctx.pipeline.loop_hooks.pre_generate {
                if lctx.cancel.is_cancelled() {
                    break 'dispatch;
                }
                let input =
                    self.build_input(lctx.execution_id, lctx.state, lctx.request, hook, None);
                hook.activity
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
                    .drain_pre_post_loop(
                        lctx.execution_id,
                        lctx.event_rx,
                        lctx.state,
                        lctx.cancel,
                    )
                    .await?;
                if let Some(err) = terminate {
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
            }

            // 2c. Record generation call against budget.
            if let Err(reason) = lctx.state.budget.record_generation() {
                self.record_event(
                    lctx.execution_id,
                    &PipelineEvent::Budget(BudgetEvent::Exhausted {
                        resource: reason.to_string(),
                    }),
                    None,
                )
                .await?;
                break 'dispatch;
            }

            // 2d. Idempotency check for generate.
            let gen_idempotency_key = format!(
                "{}:{}:{}",
                lctx.execution_id,
                lctx.pipeline.generate.activity.name(),
                lctx.state.iteration
            );

            if let Some(cached_events) = self
                .check_idempotency(lctx.execution_id, &gen_idempotency_key)
                .await?
            {
                // Replay cached events without re-running the activity.
                debug!(key = %gen_idempotency_key, "idempotency hit: replaying cached events");
                let mut commands_queued: Vec<weft_core::CommandInvocation> = Vec::new();

                for ev in cached_events {
                    if let Ok(pe) = serde_json::from_value::<PipelineEvent>(ev.payload.clone())
                        && let PipelineEvent::Generation(GenerationEvent::Chunk(ref gen_ev)) = pe
                    {
                        if let GeneratedEvent::Content { .. } = gen_ev
                            && let Some(client) = client_tx
                        {
                            let _ = client.send(pe.clone()).await;
                        }
                        match gen_ev {
                            GeneratedEvent::Content { part } => {
                                if let weft_core::ContentPart::Text(t) = part {
                                    lctx.state.accumulated_text.push_str(t);
                                }
                            }
                            GeneratedEvent::CommandInvocation(inv) => {
                                commands_queued.push(inv.clone());
                            }
                            GeneratedEvent::Done
                            | GeneratedEvent::Refused { .. }
                            | GeneratedEvent::Reasoning { .. } => {}
                        }
                    }
                }

                if commands_queued.is_empty() {
                    break 'dispatch;
                }
                let terminate = self
                    .run_commands_with_ctx(lctx, &mut commands_queued, default_command_timeout)
                    .await?;
                if let Some(err) = terminate {
                    self.finalize_failed(lctx.execution_id, lctx.state, err)
                        .await?;
                    return Err(ReactorError::Cancelled {
                        reason: "command failed".to_string(),
                    });
                }
                self.record_iteration(lctx, lctx.state.commands_executed)
                    .await?;
                continue 'dispatch;
            }

            // 2e–2f. Spawn generate and run the inner generate loop.
            let outcome = self
                .run_generate_loop(
                    lctx,
                    client_tx,
                    gen_idempotency_key,
                    default_generation_timeout,
                )
                .await?;

            match outcome {
                GenerateOutcome::Cancelled(result) => return Ok(Some(result)),
                GenerateOutcome::ChannelClosed => return Err(ReactorError::ChannelClosed),
                GenerateOutcome::BudgetExhausted => break 'dispatch,

                GenerateOutcome::ActivityFailed { retryable, error } => {
                    if retryable {
                        let policy = lctx.pipeline.generate.retry_policy.as_ref();
                        if should_retry(
                            policy,
                            lctx.state.generate_retry_attempt,
                            &lctx.state.budget,
                            lctx.cancel,
                        ) {
                            let backoff =
                                backoff_ms(policy.unwrap(), lctx.state.generate_retry_attempt);
                            let retry_event =
                                PipelineEvent::Activity(ActivityEvent::Retried {
                                    name: lctx.pipeline.generate.activity.name().to_string(),
                                    attempt: lctx.state.generate_retry_attempt + 1,
                                    error: error.clone(),
                                });
                            self.record_event(
                                lctx.execution_id,
                                &retry_event,
                                Some(&serde_json::json!({ "backoff_ms": backoff })),
                            )
                            .await?;
                            lctx.state.generate_retry_attempt += 1;

                            // Backoff with cancellation check.
                            let sleep = tokio::time::sleep(Duration::from_millis(backoff));
                            tokio::select! {
                                _ = sleep => {}
                                _ = lctx.cancel.cancelled() => {
                                    self.record_event(
                                        lctx.execution_id,
                                        &PipelineEvent::Execution(ExecutionEvent::Cancelled {
                                            reason: "cancelled during retry backoff".to_string(),
                                        }),
                                        Some(&serde_json::json!({ "partial_text": lctx.state.accumulated_text })),
                                    ).await?;
                                    self.event_log
                                        .update_execution_status(lctx.execution_id, ExecutionStatus::Cancelled)
                                        .await?;
                                    let duration_ms = lctx.state.start_time.elapsed().as_millis() as u64;
                                    let result = ExecutionResult {
                                        execution_id: lctx.execution_id.clone(),
                                        response: lctx.state.response.take().unwrap_or_else(|| empty_response(lctx.execution_id)),
                                        budget_used: BudgetUsage {
                                            generation_calls: lctx.state.budget.max_generation_calls - lctx.state.budget.remaining_generation_calls,
                                            commands_executed: lctx.state.commands_executed,
                                            iterations: lctx.state.iteration,
                                            depth_reached: lctx.state.budget.current_depth,
                                            duration_ms,
                                        },
                                        final_budget: lctx.state.budget.clone(),
                                    };
                                    return Ok(Some(result));
                                }
                            }
                            continue 'dispatch;
                        }
                    }

                    // Not retryable or exhausted retries.
                    self.record_event(
                        lctx.execution_id,
                        &PipelineEvent::Execution(ExecutionEvent::Failed {
                            error: error.clone(),
                        }),
                        Some(&serde_json::json!({ "partial_text": lctx.state.accumulated_text })),
                    )
                    .await?;
                    self.event_log
                        .update_execution_status(lctx.execution_id, ExecutionStatus::Failed)
                        .await?;
                    return Err(ReactorError::ActivityFailed(
                        crate::activity::ActivityError::Failed {
                            name: lctx.pipeline.generate.activity.name().to_string(),
                            reason: error,
                        },
                    ));
                }

                GenerateOutcome::Done {
                    mut commands_queued,
                    generation_done,
                    generation_refused,
                } => {
                    // 2g. Handle done/refused: run pre_response hooks.
                    if generation_done || generation_refused {
                        for hook in &lctx.pipeline.loop_hooks.pre_response {
                            if lctx.cancel.is_cancelled() {
                                break;
                            }
                            let input = self.build_input(
                                lctx.execution_id,
                                lctx.state,
                                lctx.request,
                                hook,
                                None,
                            );
                            hook.activity
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
                                .drain_pre_post_loop(
                                    lctx.execution_id,
                                    lctx.event_rx,
                                    lctx.state,
                                    lctx.cancel,
                                )
                                .await?;
                            if let Some(err) = terminate {
                                // Hook blocked with feedback: inject and retry generation.
                                if let ReactorError::HookBlocked { hook_name, reason } = &err {
                                    let feedback_msg = weft_core::WeftMessage {
                                        role: weft_core::Role::User,
                                        source: weft_core::Source::Client,
                                        model: None,
                                        content: vec![weft_core::ContentPart::Text(
                                            reason.clone(),
                                        )],
                                        delta: false,
                                        message_index: lctx.state.messages.len() as u32,
                                    };
                                    lctx.state.messages.push(feedback_msg.clone());
                                    self.record_event(
                                        lctx.execution_id,
                                        &PipelineEvent::Context(ContextEvent::MessageInjected {
                                            message: feedback_msg,
                                            source: MessageInjectionSource::HookFeedback {
                                                hook_name: hook_name.clone(),
                                            },
                                        }),
                                        None,
                                    )
                                    .await?;
                                    lctx.state.generate_retry_attempt = 0;
                                    continue 'dispatch;
                                }
                                // Any other error terminates.
                                self.finalize_failed(lctx.execution_id, lctx.state, err)
                                    .await?;
                                return Err(ReactorError::Cancelled {
                                    reason: "pre_response hook failed".to_string(),
                                });
                            }
                        }

                        if commands_queued.is_empty() {
                            break 'dispatch;
                        }
                    }

                    // 2h. Execute commands sequentially.
                    if !commands_queued.is_empty() {
                        let terminate = self
                            .run_commands_with_ctx(
                                lctx,
                                &mut commands_queued,
                                default_command_timeout,
                            )
                            .await?;
                        if let Some(err) = terminate {
                            self.finalize_failed(lctx.execution_id, lctx.state, err)
                                .await?;
                            return Err(ReactorError::ActivityFailed(
                                crate::activity::ActivityError::Failed {
                                    name: "execute_command".to_string(),
                                    reason: "command execution failed".to_string(),
                                },
                            ));
                        }
                    }

                    // 2i. Record iteration.
                    let cmds_this_iter = commands_queued.len() as u32;
                    self.record_iteration(lctx, cmds_this_iter).await?;
                }
            }
        } // end 'dispatch

        Ok(None)
    }

    /// Build a CommandContext from a LoopContext and run execute_commands.
    fn run_commands_with_ctx<'a>(
        &'a self,
        lctx: &'a mut LoopContext<'_>,
        commands: &'a mut Vec<weft_core::CommandInvocation>,
        default_timeout_secs: u64,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Option<ReactorError>, ReactorError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let mut cmd_ctx = CommandContext {
                execution_id: lctx.execution_id,
                state: lctx.state,
                request: lctx.request,
                pipeline: lctx.pipeline,
                event_tx: lctx.event_tx,
                event_rx: lctx.event_rx,
                cancel: lctx.cancel,
                default_timeout_secs,
            };
            self.execute_commands(&mut cmd_ctx, commands).await
        })
    }

    /// Record an iteration completion and update iteration counter.
    async fn record_iteration(
        &self,
        lctx: &mut LoopContext<'_>,
        cmds_this_iter: u32,
    ) -> Result<(), ReactorError> {
        if let Err(reason) = lctx.state.budget.record_iteration() {
            self.record_event(
                lctx.execution_id,
                &PipelineEvent::Budget(BudgetEvent::Exhausted {
                    resource: reason.to_string(),
                }),
                None,
            )
            .await?;
            // Caller will break on next budget check; just return.
            return Ok(());
        }
        self.record_event(
            lctx.execution_id,
            &PipelineEvent::Execution(ExecutionEvent::IterationCompleted {
                iteration: lctx.state.iteration,
            }),
            Some(&serde_json::json!({ "commands_executed_this_iteration": cmds_this_iter })),
        )
        .await?;
        lctx.state.iteration += 1;
        lctx.state.generate_retry_attempt = 0;
        Ok(())
    }
}
