//! The inner `'generate` select! loop and its outcome type.
//!
//! Spawns the generate activity and processes its events until the activity
//! completes, fails, times out, or the execution is cancelled. Returns a
//! `GenerateOutcome` that the `'dispatch` loop uses to decide the next step.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::warn;

use crate::error::ReactorError;
use crate::event::{
    ActivityEvent, BudgetEvent, ContextEvent, ExecutionEvent, GeneratedEvent, GenerationEvent,
    MessageInjectionSource, PipelineEvent, SignalEvent,
};
use crate::execution::ExecutionStatus;
use crate::signal::Signal;

use super::Reactor;
use super::helpers::empty_response;
use super::types::{BudgetUsage, ExecutionResult, LoopContext};

/// Outcome of running the `'generate` inner loop.
pub(super) enum GenerateOutcome {
    /// Generation finished normally. `commands_queued` may be non-empty.
    Done {
        commands_queued: Vec<weft_core::CommandInvocation>,
        generation_done: bool,
        generation_refused: bool,
    },
    /// Activity failed (after all retry decisions are left to the caller).
    ActivityFailed { retryable: bool, error: String },
    /// Budget exhausted from a BudgetExhausted event inside the generate loop.
    BudgetExhausted,
    /// Channel closed unexpectedly.
    ChannelClosed,
    /// Execution was cancelled; caller should return this result immediately.
    Cancelled(ExecutionResult),
}

impl Reactor {
    /// Spawn the generate activity and process its events.
    ///
    /// Returns a `GenerateOutcome` describing what happened.
    pub(super) async fn run_generate_loop(
        &self,
        lctx: &mut LoopContext<'_>,
        client_tx: &Option<mpsc::Sender<PipelineEvent>>,
        gen_idempotency_key: String,
        default_generation_timeout: u64,
    ) -> Result<GenerateOutcome, ReactorError> {
        let gen_input = self.build_input(
            lctx.execution_id,
            lctx.state,
            lctx.request,
            &lctx.pipeline.generate,
            Some(gen_idempotency_key),
        );

        let chunk_timeout = Duration::from_secs(
            lctx.pipeline
                .generate
                .timeout_secs
                .unwrap_or(default_generation_timeout),
        );
        let heartbeat_interval = lctx.pipeline.generate.heartbeat_interval_secs;

        // Spawn the generate activity on a separate task so we can concurrently
        // process events from it while it produces them.
        let gen_activity = Arc::clone(&lctx.pipeline.generate.activity);
        let gen_event_tx = lctx.event_tx.clone();
        let gen_exec_id = lctx.execution_id.clone();
        let gen_services = Arc::clone(&self.services);
        let gen_event_log: Arc<dyn crate::event_log::EventLog> = Arc::clone(&self.event_log);
        let gen_cancel = lctx.cancel.clone();

        let gen_handle = tokio::spawn(async move {
            gen_activity
                .execute(
                    &gen_exec_id,
                    gen_input,
                    gen_services.as_ref(),
                    gen_event_log.as_ref(),
                    gen_event_tx,
                    gen_cancel,
                )
                .await;
        });

        let mut commands_queued: Vec<weft_core::CommandInvocation> = Vec::new();
        let mut generation_done = false;
        let mut generation_refused = false;
        let mut activity_failed = false;
        let mut failed_retryable = false;
        let mut failed_error = String::new();
        let mut chunks_this_generation: u32 = 0;

        let mut chunk_deadline = tokio::time::Instant::now() + chunk_timeout;
        let mut heartbeat_expiry = heartbeat_interval
            .map(|secs| tokio::time::Instant::now() + Duration::from_secs(secs * 2));

        let current_model = lctx
            .state
            .routing
            .as_ref()
            .map(|r| r.model_routing.model.clone())
            .unwrap_or_else(|| "unknown".to_string());

        'generate: loop {
            // Compute the next deadline to sleep until.
            let sleep_until = if let Some(hb_exp) = heartbeat_expiry {
                chunk_deadline.min(hb_exp)
            } else {
                chunk_deadline
            };

            tokio::select! {
                biased;

                // Cancellation takes priority.
                _ = lctx.cancel.cancelled() => {
                    gen_handle.abort();
                    let _ = gen_handle.await;
                    tracing::debug!(execution_id = %lctx.execution_id, "cancelled during generate dispatch");
                    self.record_event(
                        lctx.execution_id,
                        &PipelineEvent::Execution(ExecutionEvent::Cancelled {
                            reason: "Signal::Cancel received".to_string(),
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
                    return Ok(GenerateOutcome::Cancelled(result));
                }

                event_opt = lctx.event_rx.recv() => {
                    match event_opt {
                        None => {
                            return Ok(GenerateOutcome::ChannelClosed);
                        }
                        Some(event) => {
                            // Record every event to the log.
                            self.record_event(lctx.execution_id, &event, None).await?;

                            // Update heartbeat/chunk timers on any event from generate.
                            match &event {
                                PipelineEvent::Generation(_)
                                | PipelineEvent::Activity(ActivityEvent::Started { .. })
                                | PipelineEvent::Activity(ActivityEvent::Completed { .. })
                                | PipelineEvent::Activity(ActivityEvent::Failed { .. }) => {
                                    chunk_deadline = tokio::time::Instant::now() + chunk_timeout;
                                    if let Some(secs) = heartbeat_interval {
                                        heartbeat_expiry = Some(tokio::time::Instant::now() + Duration::from_secs(secs * 2));
                                    }
                                }
                                PipelineEvent::Activity(ActivityEvent::Heartbeat { activity_name }) => {
                                    lctx.state.last_activity_event.insert(activity_name.clone(), std::time::Instant::now());
                                    if let Some(secs) = heartbeat_interval {
                                        heartbeat_expiry = Some(tokio::time::Instant::now() + Duration::from_secs(secs * 2));
                                    }
                                    chunk_deadline = tokio::time::Instant::now() + chunk_timeout;
                                }
                                _ => {}
                            }

                            // Forward Generation(Chunk(Content)) events to client stream.
                            if let PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content { .. })) = &event {
                                if let Some(client) = client_tx {
                                    let _ = client.send(event.clone()).await;
                                }
                                chunks_this_generation += 1;
                            }

                            // Dispatch by type.
                            match event {
                                PipelineEvent::Generation(GenerationEvent::Chunk(ref gen_ev)) => {
                                    match gen_ev {
                                        GeneratedEvent::Content { part } => {
                                            if let weft_core::ContentPart::Text(t) = part {
                                                lctx.state.accumulated_text.push_str(t);
                                            }
                                        }
                                        GeneratedEvent::CommandInvocation(inv) => {
                                            commands_queued.push(inv.clone());
                                        }
                                        GeneratedEvent::Done => {
                                            generation_done = true;
                                        }
                                        GeneratedEvent::Refused { .. } => {
                                            generation_refused = true;
                                        }
                                        GeneratedEvent::Reasoning { .. } => {
                                            // Recorded only.
                                        }
                                    }
                                }
                                PipelineEvent::Signal(SignalEvent::Received(ref signal)) => {
                                    let signal_type = signal.signal_type().to_string();
                                    let payload = serde_json::to_value(signal).unwrap_or_default();
                                    self.record_event(
                                        lctx.execution_id,
                                        &PipelineEvent::Signal(SignalEvent::Logged {
                                            signal_type,
                                            payload,
                                        }),
                                        None,
                                    ).await?;
                                    match signal.clone() {
                                        Signal::Cancel { reason } => {
                                            lctx.cancel.cancel();
                                            gen_handle.abort();
                                            let _ = gen_handle.await;
                                            self.record_event(
                                                lctx.execution_id,
                                                &PipelineEvent::Execution(ExecutionEvent::Cancelled {
                                                    reason: reason.clone(),
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
                                            return Ok(GenerateOutcome::Cancelled(result));
                                        }
                                        Signal::InjectContext { messages } => {
                                            for msg in &messages {
                                                self.record_event(
                                                    lctx.execution_id,
                                                    &PipelineEvent::Context(ContextEvent::MessageInjected {
                                                        message: msg.clone(),
                                                        source: MessageInjectionSource::SignalInjection,
                                                    }),
                                                    None,
                                                ).await?;
                                            }
                                            lctx.state.messages.extend(messages);
                                        }
                                        Signal::UpdateBudget { changes } => {
                                            lctx.state.budget.apply_update(changes);
                                        }
                                        Signal::ForceGenerationConfig { config } => {
                                            lctx.state.generation_config_override = Some(config);
                                        }
                                        Signal::Pause => {
                                            // Enter pause loop: recv until Resume or Cancel.
                                            loop {
                                                match lctx.event_rx.recv().await {
                                                    None => return Ok(GenerateOutcome::ChannelClosed),
                                                    Some(PipelineEvent::Signal(SignalEvent::Received(Signal::Resume))) => break,
                                                    Some(PipelineEvent::Signal(SignalEvent::Received(Signal::Cancel { reason }))) => {
                                                        lctx.cancel.cancel();
                                                        self.record_event(
                                                            lctx.execution_id,
                                                            &PipelineEvent::Execution(ExecutionEvent::Cancelled {
                                                                reason: reason.clone(),
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
                                                        return Ok(GenerateOutcome::Cancelled(result));
                                                    }
                                                    Some(_) => {
                                                        // Discard other events while paused.
                                                    }
                                                }
                                            }
                                        }
                                        Signal::Resume => {
                                            // Resume without prior Pause is a no-op.
                                        }
                                    }
                                }
                                PipelineEvent::Budget(BudgetEvent::Exhausted { resource }) => {
                                    tracing::debug!(resource = %resource, "budget exhausted event");
                                    lctx.cancel.cancel();
                                    gen_handle.abort();
                                    let _ = gen_handle.await;
                                    return Ok(GenerateOutcome::BudgetExhausted);
                                }
                                PipelineEvent::Activity(ActivityEvent::Completed { ref name, .. })
                                    if name == lctx.pipeline.generate.activity.name() =>
                                {
                                    // Generate activity finished — exit inner loop.
                                    break 'generate;
                                }
                                PipelineEvent::Activity(ActivityEvent::Failed { ref name, ref error, retryable })
                                    if name == lctx.pipeline.generate.activity.name() =>
                                {
                                    activity_failed = true;
                                    failed_retryable = retryable;
                                    failed_error = error.clone();
                                    break 'generate;
                                }
                                PipelineEvent::Generation(GenerationEvent::Completed {
                                    input_tokens,
                                    output_tokens,
                                    ..
                                }) => {
                                    // Accumulate token usage from each generation call.
                                    if let Some(n) = input_tokens {
                                        lctx.state.accumulated_usage.prompt_tokens += n;
                                        lctx.state.accumulated_usage.total_tokens += n;
                                    }
                                    if let Some(n) = output_tokens {
                                        lctx.state.accumulated_usage.completion_tokens += n;
                                        lctx.state.accumulated_usage.total_tokens += n;
                                    }
                                    lctx.state.accumulated_usage.llm_calls += 1;
                                }
                                _ => {
                                    // Recorded already, no state change needed.
                                }
                            }
                        }
                    }
                }

                _ = tokio::time::sleep_until(sleep_until) => {
                    // No event within timeout. Provider is hung or heartbeat missed.
                    warn!(
                        execution_id = %lctx.execution_id,
                        model = %current_model,
                        timeout_secs = chunk_timeout.as_secs(),
                        chunks_received = chunks_this_generation,
                        "generation timeout / heartbeat miss"
                    );
                    lctx.cancel.cancel();
                    gen_handle.abort();
                    let _ = gen_handle.await;

                    // Push GenerationTimedOut event.
                    let timed_out_event = PipelineEvent::Generation(GenerationEvent::TimedOut {
                        model: current_model.clone(),
                        timeout_secs: chunk_timeout.as_secs(),
                    });
                    self.record_event(
                        lctx.execution_id,
                        &timed_out_event,
                        Some(&serde_json::json!({ "chunks_received": chunks_this_generation })),
                    ).await?;

                    // Treat timeout as a retryable failure.
                    activity_failed = true;
                    failed_retryable = true;
                    failed_error = format!(
                        "generation timed out after {} chunks",
                        chunks_this_generation
                    );
                    break 'generate;
                }
            }

            // If activity failed, exit inner loop.
            if activity_failed {
                break 'generate;
            }
            // If Done/Refused, keep looping to drain remaining events including ActivityCompleted.
        }

        if activity_failed {
            return Ok(GenerateOutcome::ActivityFailed {
                retryable: failed_retryable,
                error: failed_error,
            });
        }

        Ok(GenerateOutcome::Done {
            commands_queued,
            generation_done,
            generation_refused,
        })
    }
}
