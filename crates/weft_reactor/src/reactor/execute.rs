//! Reactor::execute: setup, phase routing, and finalization.
//!
//! The top-level execute method orchestrates the three phases:
//! 1. Pre-loop activities (validate, model selection, etc.)
//! 2. Dispatch loop (generate + commands)
//! 3. Post-loop activities (assemble_response)
//!
//! Each phase is implemented in its own submodule; execute() stitches them together.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, Span, info_span};

use crate::budget::Budget;
use crate::error::ReactorError;
use crate::event::{ExecutionEvent, PipelineEvent};
use crate::execution::{Execution, ExecutionId, ExecutionStatus};

use super::Reactor;
use super::helpers::empty_response;
use super::types::{BudgetUsage, ExecutionContext, ExecutionResult, ExecutionState, LoopContext};

// Channel buffer size. Spec mandates 256; the deadlock analysis in the spec
// is based on this value (pre-loop activities bounded, generate runs async).
const CHANNEL_BUFFER: usize = 256;

impl Reactor {
    /// Run a pipeline execution.
    ///
    /// Returns the execution result and the event channel sender so that
    /// external systems can inject signals after execute returns.
    ///
    /// `cancel`: When `Some`, the execution's `CancellationToken` is
    /// created as a child of the provided token. Cancelling the parent token
    /// therefore also cancels this execution. Pass `None` for root executions.
    pub async fn execute(
        &self,
        ctx: ExecutionContext,
        cancel: Option<&CancellationToken>,
    ) -> Result<(ExecutionResult, mpsc::Sender<PipelineEvent>), ReactorError> {
        let ExecutionContext {
            request,
            tenant_id,
            request_id,
            parent_id,
            parent_budget,
            client_tx,
        } = ctx;

        // ── Setup ─────────────────────────────────────────────────────────────
        let execution_id = ExecutionId::new();
        // When a cancel token is provided (child execution), create a child
        // token so that cancelling the parent also cancels this execution.
        let cancel = match cancel {
            Some(parent) => parent.child_token(),
            None => CancellationToken::new(),
        };

        let (event_tx, mut event_rx) = mpsc::channel::<PipelineEvent>(CHANNEL_BUFFER);

        // Resolve pipeline name from request metadata or fall back to "default".
        let pipeline_name = "default";
        let pipeline = self
            .pipelines
            .get(pipeline_name)
            .ok_or_else(|| ReactorError::PipelineNotFound(pipeline_name.to_string()))?;

        // Build budget from parent or from reactor budget defaults.
        let budget = match parent_budget {
            Some(pb) => pb
                .child_budget()
                .map_err(|r| ReactorError::BudgetExhausted(r.to_string()))?,
            None => {
                let bc = &self.budget_defaults;
                Budget::new(
                    bc.max_generation_calls,
                    bc.max_iterations,
                    bc.max_depth,
                    chrono::Utc::now() + chrono::Duration::seconds(bc.timeout_secs as i64),
                )
            }
        };

        let depth = budget.current_depth;

        // ── `request` span ────────────────────────────────────────────────────
        //
        // Attributes known at creation time are recorded immediately.
        // Attributes only known at close (status, counters) use Empty sentinels
        // and are filled via Span::current().record() before the span exits.
        //
        // .instrument() is correct for async code — it restores the span across
        // .await suspension points. Never use .enter()/.guard() in async functions.
        let request_span = info_span!(
            "request",
            weft.request_id = %request_id.0,
            weft.tenant_id = %tenant_id.0,
            weft.pipeline = %pipeline_name,
            weft.depth = depth,
            weft.parent_execution_id = tracing::field::Empty,
            otel.kind = "server",
            otel.status_code = tracing::field::Empty,
            weft.generation_calls = tracing::field::Empty,
            weft.commands_executed = tracing::field::Empty,
            weft.iterations = tracing::field::Empty,
            weft.request.degraded = tracing::field::Empty,
            weft.request.degradation_count = tracing::field::Empty,
        );

        // Fill parent execution ID when this is a child execution.
        if let Some(ref pid) = parent_id {
            request_span.record("weft.parent_execution_id", pid.to_string().as_str());
        }

        async move {
            // Create execution record in EventLog.
            let execution = Execution {
                id: execution_id.clone(),
                tenant_id: tenant_id.clone(),
                request_id: request_id.clone(),
                parent_id: parent_id.clone(),
                pipeline_name: pipeline_name.to_string(),
                status: ExecutionStatus::Running,
                created_at: chrono::Utc::now(),
                depth,
            };
            self.event_log.create_execution(&execution).await?;

            // Record ExecutionStarted event.
            let budget_snapshot = budget.snapshot();
            self.record_event(
                &execution_id,
                &PipelineEvent::Execution(ExecutionEvent::Started {
                    pipeline_name: pipeline_name.to_string(),
                    tenant_id: tenant_id.0.clone(),
                    request_id: request_id.0.clone(),
                    parent_id: parent_id.as_ref().map(|id| id.to_string()),
                    depth,
                    budget: budget_snapshot,
                }),
                None,
            )
            .await?;

            // Snapshot resolved config for this request (used by degradation fallback logic).
            let resolved_config = Arc::clone(&self.services.resolved_config);
            let mut state = ExecutionState::new(budget, resolved_config);
            // Seed the message list from the incoming request.
            state.messages = request.messages.clone();

            let mut lctx = LoopContext {
                execution_id: &execution_id,
                state: &mut state,
                request: &request,
                pipeline,
                event_tx: &event_tx,
                event_rx: &mut event_rx,
                cancel: &cancel,
            };

            // ── Phase 1: Pre-loop activities ──────────────────────────────────
            if let Err(e) = self.run_pre_loop(&mut lctx).await {
                Span::current().record("otel.status_code", "ERROR");
                return Err(e);
            }

            // ── Phase 2: Dispatch loop ────────────────────────────────────────
            let default_generation_timeout = self.budget_defaults.generation_timeout_secs;
            let default_command_timeout = self.budget_defaults.command_timeout_secs;

            let dispatch_result = self
                .run_dispatch_loop(
                    &mut lctx,
                    &client_tx,
                    default_generation_timeout,
                    default_command_timeout,
                )
                .await;

            match dispatch_result {
                Err(e) => {
                    Span::current().record("otel.status_code", "ERROR");
                    return Err(e);
                }
                Ok(Some(early_result)) => {
                    // Dispatch loop returned early (cancellation).
                    Span::current().record("otel.status_code", "ERROR");
                    return Ok((early_result, event_tx));
                }
                Ok(None) => {}
            }

            // ── Phase 3: Post-loop activities ─────────────────────────────────
            if let Err(e) = self.run_post_loop(&mut lctx).await {
                Span::current().record("otel.status_code", "ERROR");
                return Err(e);
            }

            // ── Finalization ──────────────────────────────────────────────────
            let duration_ms = lctx.state.start_time.elapsed().as_millis() as u64;
            let generation_calls_used = lctx.state.budget.max_generation_calls
                - lctx.state.budget.remaining_generation_calls;

            // Record close-time span attributes before the span exits.
            Span::current().record("weft.generation_calls", generation_calls_used);
            Span::current().record("weft.commands_executed", lctx.state.commands_executed);
            Span::current().record("weft.iterations", lctx.state.iteration);
            Span::current().record("otel.status_code", "OK");

            // Record degradation attributes on the request span.
            // The request span stays OK (degraded != failed), but operators
            // can filter for degraded requests via these attributes.
            let degradation_count = lctx.state.degradations.len();
            if degradation_count > 0 {
                Span::current().record("weft.request.degraded", true);
                Span::current().record("weft.request.degradation_count", degradation_count as u64);
            } else {
                Span::current().record("weft.request.degraded", false);
                Span::current().record("weft.request.degradation_count", 0u64);
            }

            // ExecutionEvent::Completed is now a unit variant (Phase 2 slimming).
            // Observability fields are preserved as _obs_* enrichment in the stored payload.
            self.record_event(
                &execution_id,
                &PipelineEvent::Execution(ExecutionEvent::Completed),
                Some(&serde_json::json!({
                    "generation_calls": generation_calls_used,
                    "commands_executed": lctx.state.commands_executed,
                    "iterations": lctx.state.iteration,
                    "duration_ms": duration_ms,
                })),
            )
            .await?;
            self.event_log
                .update_execution_status(&execution_id, ExecutionStatus::Completed)
                .await?;

            let response = lctx
                .state
                .response
                .take()
                .unwrap_or_else(|| empty_response(&execution_id));

            // Take degradations via mem::take to avoid cloning the Vec.
            let degradations = std::mem::take(&mut lctx.state.degradations);

            let result = ExecutionResult {
                execution_id: execution_id.clone(),
                response,
                budget_used: BudgetUsage {
                    generation_calls: generation_calls_used,
                    commands_executed: lctx.state.commands_executed,
                    iterations: lctx.state.iteration,
                    depth_reached: lctx.state.budget.current_depth,
                    duration_ms,
                },
                final_budget: lctx.state.budget.clone(),
                degradations,
            };

            Ok((result, event_tx))
        }
        .instrument(request_span)
        .await
    }
}
