//! Reactor: dispatch loop consuming events from a channel.
//!
//! The Reactor is the execution engine for the Weft pipeline. It is NOT a
//! step-by-step executor. It is an event dispatch loop:
//!
//! 1. Run pre-loop activities sequentially (validate, route, assemble_prompt).
//! 2. Enter the dispatch loop:
//!    a. Spawn the generate activity as an event producer.
//!    b. Receive events from the channel via `tokio::select!`.
//!    c. Dispatch by variant (Generated, CommandInvocation, Signal, etc.).
//!    d. After commands complete, loop back to generation.
//! 3. Run post-loop activities sequentially (assemble_response).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::activity::{Activity, ActivityInput, RoutingSnapshot};
use crate::budget::{Budget, BudgetCheck};
use crate::config::{PipelineConfig, ReactorConfig, RetryPolicy};
use crate::error::ReactorError;
use crate::event::{EVENT_SCHEMA_VERSION, Event, GeneratedEvent, PipelineEvent};
use crate::event_log::EventLog;
use crate::execution::{Execution, ExecutionId, ExecutionStatus, RequestId, TenantId};
use crate::registry::ActivityRegistry;
use crate::services::Services;
use crate::signal::Signal;

// Channel buffer size. Spec mandates 256; the deadlock analysis in the spec
// is based on this value (pre-loop activities bounded, generate runs async).
const CHANNEL_BUFFER: usize = 256;

// ── Public types ──────────────────────────────────────────────────────────────

/// The execution engine. A dispatch loop consuming events from a channel.
pub struct Reactor {
    services: Arc<Services>,
    event_log: Arc<dyn EventLog>,
    #[allow(dead_code)]
    registry: Arc<ActivityRegistry>,
    /// Compiled pipelines: activity references resolved to Arc<dyn Activity>.
    pipelines: HashMap<String, CompiledPipeline>,
    /// Default budget settings from ReactorConfig.
    budget_defaults: crate::config::BudgetConfig,
}

/// Result of executing a pipeline.
#[derive(Debug)]
pub struct ExecutionResult {
    pub execution_id: ExecutionId,
    pub response: weft_core::WeftResponse,
    pub budget_used: BudgetUsage,
    /// The execution's final budget state. Used by spawn_child to deduct
    /// child consumption from the parent via deduct_child_usage.
    pub final_budget: Budget,
}

/// Summary of resources consumed by an execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BudgetUsage {
    pub generation_calls: u32,
    pub commands_executed: u32,
    pub iterations: u32,
    pub depth_reached: u32,
    pub duration_ms: u64,
}

// ── Private types ─────────────────────────────────────────────────────────────

/// A pipeline with all activity references resolved.
struct CompiledPipeline {
    #[allow(dead_code)]
    config: PipelineConfig,
    pre_loop: Vec<ResolvedActivity>,
    post_loop: Vec<ResolvedActivity>,
    generate: ResolvedActivity,
    execute_command: ResolvedActivity,
    loop_hooks: CompiledLoopHooks,
}

/// An activity reference resolved from the registry, carrying its
/// runtime config (retry policy, timeout, heartbeat interval).
struct ResolvedActivity {
    activity: Arc<dyn Activity>,
    retry_policy: Option<RetryPolicy>,
    timeout_secs: Option<u64>,
    heartbeat_interval_secs: Option<u64>,
}

struct CompiledLoopHooks {
    pre_generate: Vec<ResolvedActivity>,
    pre_response: Vec<ResolvedActivity>,
    pre_tool_use: Vec<ResolvedActivity>,
    post_tool_use: Vec<ResolvedActivity>,
}

/// Internal state maintained by the Reactor during execution.
/// Not passed to activities — they receive ActivityInput snapshots.
struct ExecutionState {
    /// Working message list.
    messages: Vec<weft_core::WeftMessage>,
    /// Current budget state.
    budget: Budget,
    /// Routing result, set from RouteCompleted events.
    routing: Option<RoutingSnapshot>,
    /// Generation config override from ForceGenerationConfig signal.
    generation_config_override: Option<serde_json::Value>,
    /// Accumulated text content across all generation calls.
    accumulated_text: String,
    /// Available commands, populated by ValidateActivity.
    available_commands: Vec<weft_core::CommandStub>,
    /// The assembled response, set from ResponseAssembled events.
    response: Option<weft_core::WeftResponse>,
    /// Count of commands executed across all iterations.
    commands_executed: u32,
    /// Execution start time for duration tracking.
    start_time: Instant,
    /// Current dispatch loop iteration (0-indexed).
    iteration: u32,
    /// Last event timestamp per spawned activity, for heartbeat tracking.
    last_activity_event: HashMap<String, Instant>,
    /// Current retry attempt for the generate activity (0 = initial attempt).
    generate_retry_attempt: u32,
    /// Accumulated token usage across all generation calls.
    accumulated_usage: weft_core::WeftUsage,
}

impl ExecutionState {
    fn new(budget: Budget) -> Self {
        Self {
            messages: Vec::new(),
            budget,
            routing: None,
            generation_config_override: None,
            accumulated_text: String::new(),
            available_commands: Vec::new(),
            response: None,
            commands_executed: 0,
            start_time: Instant::now(),
            iteration: 0,
            last_activity_event: HashMap::new(),
            generate_retry_attempt: 0,
            accumulated_usage: weft_core::WeftUsage::default(),
        }
    }
}

impl std::fmt::Debug for Reactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reactor")
            .field("pipelines", &self.pipelines.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ── Reactor construction ──────────────────────────────────────────────────────

impl Reactor {
    /// Build a Reactor from config.
    ///
    /// Resolves all activity name references to Arc<dyn Activity>.
    /// Returns ReactorError::ActivityNotFound if any activity name
    /// in any pipeline config is not in the registry.
    pub fn new(
        services: Arc<Services>,
        event_log: Arc<dyn EventLog>,
        registry: Arc<ActivityRegistry>,
        config: &ReactorConfig,
    ) -> Result<Self, ReactorError> {
        let mut pipelines = HashMap::new();
        for pipeline_config in &config.pipelines {
            let compiled = Self::compile_pipeline(pipeline_config, &registry)?;
            pipelines.insert(pipeline_config.name.clone(), compiled);
        }

        if !pipelines.contains_key("default") {
            return Err(ReactorError::Config(
                "no 'default' pipeline defined".to_string(),
            ));
        }

        Ok(Self {
            services,
            event_log,
            registry,
            pipelines,
            budget_defaults: config.budget.clone(),
        })
    }

    fn compile_pipeline(
        config: &PipelineConfig,
        registry: &ActivityRegistry,
    ) -> Result<CompiledPipeline, ReactorError> {
        let resolve =
            |activity_ref: &crate::config::ActivityRef| -> Result<ResolvedActivity, ReactorError> {
                let name = activity_ref.name();
                let activity = registry
                    .get(name)
                    .ok_or_else(|| ReactorError::ActivityNotFound(name.to_string()))?
                    .clone();
                Ok(ResolvedActivity {
                    activity,
                    retry_policy: activity_ref.retry_policy().cloned(),
                    timeout_secs: activity_ref.timeout_secs(),
                    heartbeat_interval_secs: activity_ref.heartbeat_interval_secs(),
                })
            };

        let pre_loop = config
            .pre_loop
            .iter()
            .map(resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let post_loop = config
            .post_loop
            .iter()
            .map(resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let generate = resolve(&config.generate)?;
        let execute_command = resolve(&config.execute_command)?;

        let loop_hooks = CompiledLoopHooks {
            pre_generate: config
                .loop_hooks
                .pre_generate
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
            pre_response: config
                .loop_hooks
                .pre_response
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
            pre_tool_use: config
                .loop_hooks
                .pre_tool_use
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
            post_tool_use: config
                .loop_hooks
                .post_tool_use
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
        };

        Ok(CompiledPipeline {
            config: config.clone(),
            pre_loop,
            post_loop,
            generate,
            execute_command,
            loop_hooks,
        })
    }
}

// ── Execute ───────────────────────────────────────────────────────────────────

impl Reactor {
    /// Run a pipeline execution.
    ///
    /// Returns the execution result and the event channel sender so that
    /// external systems can inject signals after execute returns.
    ///
    /// `parent_cancel`: When `Some`, the execution's `CancellationToken` is
    /// created as a child of the provided token. Cancelling the parent token
    /// therefore also cancels this execution. Pass `None` for root executions.
    // Spec-mandated signature requires 8 positional parameters; grouping into
    // a builder would change the public API contract. Allow lint locally.
    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        request: weft_core::WeftRequest,
        tenant_id: TenantId,
        request_id: RequestId,
        parent_id: Option<ExecutionId>,
        parent_budget: Option<Budget>,
        client_tx: Option<mpsc::Sender<PipelineEvent>>,
        parent_cancel: Option<&CancellationToken>,
    ) -> Result<(ExecutionResult, mpsc::Sender<PipelineEvent>), ReactorError> {
        // ── Setup ─────────────────────────────────────────────────────────────
        let execution_id = ExecutionId::new();
        // When a parent cancel token is provided (child execution), create a
        // child token so that cancelling the parent also cancels this execution.
        let cancel = match parent_cancel {
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
            &PipelineEvent::ExecutionStarted {
                pipeline_name: pipeline_name.to_string(),
                tenant_id: tenant_id.0.clone(),
                request_id: request_id.0.clone(),
                parent_id: parent_id.as_ref().map(|id| id.to_string()),
                depth,
                budget: budget_snapshot,
            },
        )
        .await?;

        let mut state = ExecutionState::new(budget);
        // Seed the message list from the incoming request.
        state.messages = request.messages.clone();

        // ── Phase 1: Pre-loop activities ──────────────────────────────────────
        for resolved in &pipeline.pre_loop {
            if cancel.is_cancelled() {
                break;
            }

            let input = self.build_input(&execution_id, &state, &request, resolved, None);
            resolved
                .activity
                .execute(
                    &execution_id,
                    input,
                    &self.services,
                    self.event_log.as_ref(),
                    event_tx.clone(),
                    cancel.clone(),
                )
                .await;

            // Drain and dispatch all events the activity pushed.
            let terminate = self
                .drain_pre_post_loop(&execution_id, &mut event_rx, &mut state, &cancel)
                .await?;
            if let Some(err) = terminate {
                self.record_event(
                    &execution_id,
                    &PipelineEvent::ExecutionFailed {
                        error: err.to_string(),
                        partial_text: Some(state.accumulated_text.clone()),
                    },
                )
                .await?;
                self.event_log
                    .update_execution_status(&execution_id, ExecutionStatus::Failed)
                    .await?;
                return Err(err);
            }
        }

        // ── Phase 2: Dispatch loop ────────────────────────────────────────────
        let default_generation_timeout = self.budget_defaults.generation_timeout_secs;
        let default_command_timeout = self.budget_defaults.command_timeout_secs;

        'dispatch: loop {
            // 2a. Budget check.
            match state.budget.check() {
                BudgetCheck::Exhausted(reason) => {
                    let resource = reason.to_string();
                    self.record_event(
                        &execution_id,
                        &PipelineEvent::BudgetExhausted {
                            resource: resource.clone(),
                        },
                    )
                    .await?;
                    debug!("budget exhausted: {resource}");
                    break 'dispatch;
                }
                BudgetCheck::Warning(info) => {
                    self.record_event(
                        &execution_id,
                        &PipelineEvent::BudgetWarning {
                            resource: info.resource.clone(),
                            remaining: info.remaining,
                        },
                    )
                    .await?;
                }
                BudgetCheck::Ok => {}
            }

            // 2b. Run pre_generate hooks.
            for hook in &pipeline.loop_hooks.pre_generate {
                if cancel.is_cancelled() {
                    break 'dispatch;
                }
                let input = self.build_input(&execution_id, &state, &request, hook, None);
                hook.activity
                    .execute(
                        &execution_id,
                        input,
                        &self.services,
                        self.event_log.as_ref(),
                        event_tx.clone(),
                        cancel.clone(),
                    )
                    .await;
                let terminate = self
                    .drain_pre_post_loop(&execution_id, &mut event_rx, &mut state, &cancel)
                    .await?;
                if let Some(err) = terminate {
                    self.record_event(
                        &execution_id,
                        &PipelineEvent::ExecutionFailed {
                            error: err.to_string(),
                            partial_text: Some(state.accumulated_text.clone()),
                        },
                    )
                    .await?;
                    self.event_log
                        .update_execution_status(&execution_id, ExecutionStatus::Failed)
                        .await?;
                    return Err(err);
                }
            }

            // 2c. Record generation call against budget.
            if let Err(reason) = state.budget.record_generation() {
                self.record_event(
                    &execution_id,
                    &PipelineEvent::BudgetExhausted {
                        resource: reason.to_string(),
                    },
                )
                .await?;
                break 'dispatch;
            }

            // 2d. Idempotency check for generate.
            let gen_idempotency_key = format!(
                "{}:{}:{}",
                execution_id,
                pipeline.generate.activity.name(),
                state.iteration
            );

            if let Some(cached_events) = self
                .check_idempotency(&execution_id, &gen_idempotency_key)
                .await?
            {
                // Replay cached events without re-running the activity.
                debug!(key = %gen_idempotency_key, "idempotency hit: replaying cached events");
                let mut commands_queued: Vec<weft_core::CommandInvocation> = Vec::new();
                let mut generation_done = false;
                let mut generation_refused = false;

                for ev in cached_events {
                    // Deserialize the stored payload back to PipelineEvent for dispatch.
                    if let Ok(pe) = serde_json::from_value::<PipelineEvent>(ev.payload.clone()) {
                        self.dispatch_generated_event(
                            &execution_id,
                            &pe,
                            &mut state,
                            &client_tx,
                            &mut commands_queued,
                            &mut generation_done,
                            &mut generation_refused,
                        )
                        .await?;
                    }
                }

                // Skip to command execution.
                if commands_queued.is_empty() {
                    break 'dispatch;
                }
                let terminate = self
                    .execute_commands(
                        &execution_id,
                        &mut commands_queued,
                        &mut state,
                        &request,
                        pipeline,
                        &event_tx,
                        &mut event_rx,
                        &cancel,
                        default_command_timeout,
                    )
                    .await?;
                if let Some(err) = terminate {
                    self.finalize_failed(&execution_id, &mut state, err).await?;
                    return Err(ReactorError::Cancelled {
                        reason: "command failed".to_string(),
                    });
                }

                // Record iteration.
                let cmds_this_iter = state.commands_executed;
                if let Err(reason) = state.budget.record_iteration() {
                    self.record_event(
                        &execution_id,
                        &PipelineEvent::BudgetExhausted {
                            resource: reason.to_string(),
                        },
                    )
                    .await?;
                    break 'dispatch;
                }
                self.record_event(
                    &execution_id,
                    &PipelineEvent::IterationCompleted {
                        iteration: state.iteration,
                        commands_executed_this_iteration: cmds_this_iter,
                    },
                )
                .await?;
                state.iteration += 1;
                state.generate_retry_attempt = 0;
                continue 'dispatch;
            }

            // 2e. Spawn generate activity and run the dispatch loop over its events.
            let gen_input = self.build_input(
                &execution_id,
                &state,
                &request,
                &pipeline.generate,
                Some(gen_idempotency_key.clone()),
            );

            let chunk_timeout = Duration::from_secs(
                pipeline
                    .generate
                    .timeout_secs
                    .unwrap_or(default_generation_timeout),
            );
            let heartbeat_interval = pipeline.generate.heartbeat_interval_secs;

            // Spawn the generate activity on a separate task so we can concurrently
            // process events from it while it produces them.
            let gen_activity = Arc::clone(&pipeline.generate.activity);
            let gen_event_tx = event_tx.clone();
            let gen_exec_id = execution_id.clone();
            let gen_services = Arc::clone(&self.services);
            let gen_event_log: Arc<dyn EventLog> = Arc::clone(&self.event_log);
            let gen_cancel = cancel.clone();

            let gen_handle = tokio::spawn(async move {
                gen_activity
                    .execute(
                        &gen_exec_id,
                        gen_input,
                        &gen_services,
                        gen_event_log.as_ref(),
                        gen_event_tx,
                        gen_cancel,
                    )
                    .await;
            });

            // 2f. Receive events from channel via tokio::select! with timeout/heartbeat.
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

            let current_model = state
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
                    _ = cancel.cancelled() => {
                        gen_handle.abort();
                        let _ = gen_handle.await;
                        debug!(execution_id = %execution_id, "cancelled during generate dispatch");
                        self.record_event(
                            &execution_id,
                            &PipelineEvent::ExecutionCancelled {
                                reason: "Signal::Cancel received".to_string(),
                                partial_text: Some(state.accumulated_text.clone()),
                            },
                        ).await?;
                        self.event_log
                            .update_execution_status(&execution_id, ExecutionStatus::Cancelled)
                            .await?;
                        let duration_ms = state.start_time.elapsed().as_millis() as u64;
                        let result = ExecutionResult {
                            execution_id: execution_id.clone(),
                            response: state.response.take().unwrap_or_else(|| empty_response(&execution_id)),
                            budget_used: BudgetUsage {
                                generation_calls: state.budget.max_generation_calls - state.budget.remaining_generation_calls,
                                commands_executed: state.commands_executed,
                                iterations: state.iteration,
                                depth_reached: state.budget.current_depth,
                                duration_ms,
                            },
                            final_budget: state.budget,
                        };
                        return Ok((result, event_tx));
                    }

                    event_opt = event_rx.recv() => {
                        match event_opt {
                            None => {
                                // Channel closed unexpectedly.
                                return Err(ReactorError::ChannelClosed);
                            }
                            Some(event) => {
                                // Record every event to the log.
                                self.record_event(&execution_id, &event).await?;

                                // Update heartbeat/chunk timers on any event from generate.
                                match &event {
                                    PipelineEvent::Generated(_)
                                    | PipelineEvent::GenerationStarted { .. }
                                    | PipelineEvent::GenerationCompleted { .. }
                                    | PipelineEvent::GenerationFailed { .. }
                                    | PipelineEvent::ActivityStarted { .. }
                                    | PipelineEvent::ActivityCompleted { .. }
                                    | PipelineEvent::ActivityFailed { .. } => {
                                        chunk_deadline = tokio::time::Instant::now() + chunk_timeout;
                                        if let Some(secs) = heartbeat_interval {
                                            heartbeat_expiry = Some(tokio::time::Instant::now() + Duration::from_secs(secs * 2));
                                        }
                                    }
                                    PipelineEvent::Heartbeat { activity_name } => {
                                        state.last_activity_event.insert(activity_name.clone(), Instant::now());
                                        if let Some(secs) = heartbeat_interval {
                                            heartbeat_expiry = Some(tokio::time::Instant::now() + Duration::from_secs(secs * 2));
                                        }
                                        chunk_deadline = tokio::time::Instant::now() + chunk_timeout;
                                    }
                                    _ => {}
                                }

                                // Forward Generated(Content) events to client stream.
                                if let PipelineEvent::Generated(GeneratedEvent::Content { .. }) = &event {
                                    if let Some(ref client) = client_tx {
                                        let _ = client.send(event.clone()).await;
                                    }
                                    chunks_this_generation += 1;
                                }

                                // Dispatch by type.
                                match event {
                                    PipelineEvent::Generated(ref gen_ev) => {
                                        match gen_ev {
                                            GeneratedEvent::Content { part } => {
                                                if let weft_core::ContentPart::Text(t) = part {
                                                    state.accumulated_text.push_str(t);
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
                                    PipelineEvent::Signal(ref signal) => {
                                        let signal_type = signal.signal_type().to_string();
                                        let payload = serde_json::to_value(signal).unwrap_or_default();
                                        self.record_event(
                                            &execution_id,
                                            &PipelineEvent::SignalReceived {
                                                signal_type,
                                                payload,
                                            },
                                        ).await?;
                                        match signal.clone() {
                                            Signal::Cancel { reason } => {
                                                cancel.cancel();
                                                gen_handle.abort();
                                                let _ = gen_handle.await;
                                                self.record_event(
                                                    &execution_id,
                                                    &PipelineEvent::ExecutionCancelled {
                                                        reason: reason.clone(),
                                                        partial_text: Some(state.accumulated_text.clone()),
                                                    },
                                                ).await?;
                                                self.event_log
                                                    .update_execution_status(&execution_id, ExecutionStatus::Cancelled)
                                                    .await?;
                                                let duration_ms = state.start_time.elapsed().as_millis() as u64;
                                                let result = ExecutionResult {
                                                    execution_id: execution_id.clone(),
                                                    response: state.response.take().unwrap_or_else(|| empty_response(&execution_id)),
                                                    budget_used: BudgetUsage {
                                                        generation_calls: state.budget.max_generation_calls - state.budget.remaining_generation_calls,
                                                        commands_executed: state.commands_executed,
                                                        iterations: state.iteration,
                                                        depth_reached: state.budget.current_depth,
                                                        duration_ms,
                                                    },
                                                    final_budget: state.budget,
                                                };
                                                return Ok((result, event_tx));
                                            }
                                            Signal::InjectContext { messages } => {
                                                state.messages.extend(messages);
                                            }
                                            Signal::UpdateBudget { changes } => {
                                                state.budget.apply_update(changes);
                                            }
                                            Signal::ForceGenerationConfig { config } => {
                                                state.generation_config_override = Some(config);
                                            }
                                            Signal::Pause => {
                                                // Enter pause loop: recv until Resume or Cancel.
                                                loop {
                                                    match event_rx.recv().await {
                                                        None => return Err(ReactorError::ChannelClosed),
                                                        Some(PipelineEvent::Signal(Signal::Resume)) => break,
                                                        Some(PipelineEvent::Signal(Signal::Cancel { reason })) => {
                                                            cancel.cancel();
                                                            self.record_event(
                                                                &execution_id,
                                                                &PipelineEvent::ExecutionCancelled {
                                                                    reason: reason.clone(),
                                                                    partial_text: Some(state.accumulated_text.clone()),
                                                                },
                                                            ).await?;
                                                            self.event_log
                                                                .update_execution_status(&execution_id, ExecutionStatus::Cancelled)
                                                                .await?;
                                                            let duration_ms = state.start_time.elapsed().as_millis() as u64;
                                                            let result = ExecutionResult {
                                                                execution_id: execution_id.clone(),
                                                                response: state.response.take().unwrap_or_else(|| empty_response(&execution_id)),
                                                                budget_used: BudgetUsage {
                                                                    generation_calls: state.budget.max_generation_calls - state.budget.remaining_generation_calls,
                                                                    commands_executed: state.commands_executed,
                                                                    iterations: state.iteration,
                                                                    depth_reached: state.budget.current_depth,
                                                                    duration_ms,
                                                                },
                                                                final_budget: state.budget,
                                                            };
                                                            return Ok((result, event_tx));
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
                                    PipelineEvent::BudgetExhausted { resource } => {
                                        debug!(resource = %resource, "budget exhausted event");
                                        cancel.cancel();
                                        gen_handle.abort();
                                        let _ = gen_handle.await;
                                        break 'dispatch;
                                    }
                                    PipelineEvent::ActivityCompleted { ref name, .. }
                                        if name == pipeline.generate.activity.name() =>
                                    {
                                        // Generate activity finished — exit inner loop.
                                        break 'generate;
                                    }
                                    PipelineEvent::ActivityFailed { ref name, ref error, retryable }
                                        if name == pipeline.generate.activity.name() =>
                                    {
                                        activity_failed = true;
                                        failed_retryable = retryable;
                                        failed_error = error.clone();
                                        break 'generate;
                                    }
                                    PipelineEvent::GenerationCompleted {
                                        input_tokens,
                                        output_tokens,
                                        ..
                                    } => {
                                        // Accumulate token usage from each generation call.
                                        if let Some(n) = input_tokens {
                                            state.accumulated_usage.prompt_tokens += n;
                                            state.accumulated_usage.total_tokens += n;
                                        }
                                        if let Some(n) = output_tokens {
                                            state.accumulated_usage.completion_tokens += n;
                                            state.accumulated_usage.total_tokens += n;
                                        }
                                        state.accumulated_usage.llm_calls += 1;
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
                            execution_id = %execution_id,
                            model = %current_model,
                            timeout_secs = chunk_timeout.as_secs(),
                            chunks_received = chunks_this_generation,
                            "generation timeout / heartbeat miss"
                        );
                        cancel.cancel();
                        gen_handle.abort();
                        let _ = gen_handle.await;

                        // Push GenerationTimedOut event.
                        let timed_out_event = PipelineEvent::GenerationTimedOut {
                            model: current_model.clone(),
                            timeout_secs: chunk_timeout.as_secs(),
                            chunks_received: chunks_this_generation,
                        };
                        self.record_event(&execution_id, &timed_out_event).await?;

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

            // ── Handle activity failure / retry (2g') ─────────────────────────
            if activity_failed {
                if failed_retryable {
                    let policy = pipeline.generate.retry_policy.as_ref();
                    if should_retry(policy, state.generate_retry_attempt, &state.budget, &cancel) {
                        let backoff = backoff_ms(policy.unwrap(), state.generate_retry_attempt);
                        let retry_event = PipelineEvent::ActivityRetried {
                            name: pipeline.generate.activity.name().to_string(),
                            attempt: state.generate_retry_attempt + 1,
                            backoff_ms: backoff,
                            error: failed_error.clone(),
                        };
                        self.record_event(&execution_id, &retry_event).await?;
                        state.generate_retry_attempt += 1;

                        // Backoff with cancellation check.
                        let sleep = tokio::time::sleep(Duration::from_millis(backoff));
                        tokio::select! {
                            _ = sleep => {}
                            _ = cancel.cancelled() => {
                                self.record_event(
                                    &execution_id,
                                    &PipelineEvent::ExecutionCancelled {
                                        reason: "cancelled during retry backoff".to_string(),
                                        partial_text: Some(state.accumulated_text.clone()),
                                    },
                                ).await?;
                                self.event_log
                                    .update_execution_status(&execution_id, ExecutionStatus::Cancelled)
                                    .await?;
                                let duration_ms = state.start_time.elapsed().as_millis() as u64;
                                let result = ExecutionResult {
                                    execution_id: execution_id.clone(),
                                    response: state.response.take().unwrap_or_else(|| empty_response(&execution_id)),
                                    budget_used: BudgetUsage {
                                        generation_calls: state.budget.max_generation_calls - state.budget.remaining_generation_calls,
                                        commands_executed: state.commands_executed,
                                        iterations: state.iteration,
                                        depth_reached: state.budget.current_depth,
                                        duration_ms,
                                    },
                                    final_budget: state.budget,
                                };
                                return Ok((result, event_tx));
                            }
                        }
                        // Reset cancel token (re-use same loop iteration).
                        // Go back to 2d (re-check idempotency, re-spawn).
                        continue 'dispatch;
                    }
                }

                // Not retryable or exhausted retries.
                let err_event = PipelineEvent::ExecutionFailed {
                    error: failed_error.clone(),
                    partial_text: Some(state.accumulated_text.clone()),
                };
                self.record_event(&execution_id, &err_event).await?;
                self.event_log
                    .update_execution_status(&execution_id, ExecutionStatus::Failed)
                    .await?;
                return Err(ReactorError::ActivityFailed(
                    crate::activity::ActivityError::Failed {
                        name: pipeline.generate.activity.name().to_string(),
                        reason: failed_error,
                    },
                ));
            }

            // ── 2g: Handle done/refused: run pre_response hooks ───────────────
            if generation_done || generation_refused {
                // Run pre_response hooks.
                for hook in &pipeline.loop_hooks.pre_response {
                    if cancel.is_cancelled() {
                        break;
                    }
                    let input = self.build_input(&execution_id, &state, &request, hook, None);
                    hook.activity
                        .execute(
                            &execution_id,
                            input,
                            &self.services,
                            self.event_log.as_ref(),
                            event_tx.clone(),
                            cancel.clone(),
                        )
                        .await;
                    let terminate = self
                        .drain_pre_post_loop(&execution_id, &mut event_rx, &mut state, &cancel)
                        .await?;
                    if let Some(err) = terminate {
                        // Hook blocked with feedback: inject and retry generation.
                        if let ReactorError::HookBlocked { reason, .. } = &err {
                            // Inject feedback message and continue dispatch loop.
                            let feedback_msg = weft_core::WeftMessage {
                                role: weft_core::Role::User,
                                source: weft_core::Source::Client,
                                model: None,
                                content: vec![weft_core::ContentPart::Text(reason.clone())],
                                delta: false,
                                message_index: state.messages.len() as u32,
                            };
                            state.messages.push(feedback_msg);
                            state.generate_retry_attempt = 0;
                            continue 'dispatch;
                        }
                        // Any other error terminates.
                        self.finalize_failed(&execution_id, &mut state, err).await?;
                        return Err(ReactorError::Cancelled {
                            reason: "pre_response hook failed".to_string(),
                        });
                    }
                }

                if commands_queued.is_empty() {
                    // Pure Done/Refused with no commands: exit dispatch loop.
                    break 'dispatch;
                }
            }

            // ── 2h: Execute commands sequentially ────────────────────────────
            if !commands_queued.is_empty() {
                let terminate = self
                    .execute_commands(
                        &execution_id,
                        &mut commands_queued,
                        &mut state,
                        &request,
                        pipeline,
                        &event_tx,
                        &mut event_rx,
                        &cancel,
                        default_command_timeout,
                    )
                    .await?;
                if let Some(err) = terminate {
                    self.finalize_failed(&execution_id, &mut state, err).await?;
                    return Err(ReactorError::ActivityFailed(
                        crate::activity::ActivityError::Failed {
                            name: "execute_command".to_string(),
                            reason: "command execution failed".to_string(),
                        },
                    ));
                }
            }

            // ── 2i: Record iteration ──────────────────────────────────────────
            let cmds_this_iter = commands_queued.len() as u32;
            if let Err(reason) = state.budget.record_iteration() {
                self.record_event(
                    &execution_id,
                    &PipelineEvent::BudgetExhausted {
                        resource: reason.to_string(),
                    },
                )
                .await?;
                break 'dispatch;
            }
            self.record_event(
                &execution_id,
                &PipelineEvent::IterationCompleted {
                    iteration: state.iteration,
                    commands_executed_this_iteration: cmds_this_iter,
                },
            )
            .await?;
            state.iteration += 1;
            state.generate_retry_attempt = 0;
        } // end 'dispatch

        // ── Phase 3: Post-loop activities ─────────────────────────────────────
        for resolved in &pipeline.post_loop {
            if cancel.is_cancelled() {
                break;
            }
            let input = self.build_input(&execution_id, &state, &request, resolved, None);
            resolved
                .activity
                .execute(
                    &execution_id,
                    input,
                    &self.services,
                    self.event_log.as_ref(),
                    event_tx.clone(),
                    cancel.clone(),
                )
                .await;
            let terminate = self
                .drain_pre_post_loop(&execution_id, &mut event_rx, &mut state, &cancel)
                .await?;
            if let Some(err) = terminate {
                // Post-loop failure: record but still return partial results.
                warn!("post-loop activity failed: {err}");
                break;
            }
        }

        // ── Finalization ──────────────────────────────────────────────────────
        let duration_ms = state.start_time.elapsed().as_millis() as u64;
        let generation_calls_used =
            state.budget.max_generation_calls - state.budget.remaining_generation_calls;

        self.record_event(
            &execution_id,
            &PipelineEvent::ExecutionCompleted {
                generation_calls: generation_calls_used,
                commands_executed: state.commands_executed,
                iterations: state.iteration,
                duration_ms,
            },
        )
        .await?;
        self.event_log
            .update_execution_status(&execution_id, ExecutionStatus::Completed)
            .await?;

        let response = state
            .response
            .take()
            .unwrap_or_else(|| empty_response(&execution_id));

        let result = ExecutionResult {
            execution_id,
            response,
            budget_used: BudgetUsage {
                generation_calls: generation_calls_used,
                commands_executed: state.commands_executed,
                iterations: state.iteration,
                depth_reached: state.budget.current_depth,
                duration_ms,
            },
            final_budget: state.budget,
        };

        Ok((result, event_tx))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

impl Reactor {
    /// Build an ActivityInput snapshot from current ExecutionState.
    ///
    /// The `execution_id` parameter is unused here but kept for future use
    /// (e.g., fetching per-execution config overrides).
    fn build_input(
        &self,
        _execution_id: &ExecutionId,
        state: &ExecutionState,
        request: &weft_core::WeftRequest,
        _resolved: &ResolvedActivity,
        idempotency_key: Option<String>,
    ) -> ActivityInput {
        // Per spec Section 6.4: generation_config derives from override, then routing.
        let generation_config = state.generation_config_override.clone().or_else(|| {
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
        }
    }

    /// Drain the channel after a synchronous pre/post-loop activity completes.
    ///
    /// Returns Some(err) if execution should terminate (ActivityFailed or HookBlocked),
    /// None if execution should continue.
    async fn drain_pre_post_loop(
        &self,
        execution_id: &ExecutionId,
        event_rx: &mut mpsc::Receiver<PipelineEvent>,
        state: &mut ExecutionState,
        cancel: &CancellationToken,
    ) -> Result<Option<ReactorError>, ReactorError> {
        loop {
            match event_rx.try_recv() {
                Ok(event) => {
                    self.record_event(execution_id, &event).await?;
                    match &event {
                        PipelineEvent::ActivityFailed { name, error, .. } => {
                            return Ok(Some(ReactorError::ActivityFailed(
                                crate::activity::ActivityError::Failed {
                                    name: name.clone(),
                                    reason: error.clone(),
                                },
                            )));
                        }
                        PipelineEvent::HookBlocked {
                            hook_name, reason, ..
                        } => {
                            return Ok(Some(ReactorError::HookBlocked {
                                hook_name: hook_name.clone(),
                                reason: reason.clone(),
                            }));
                        }
                        PipelineEvent::RouteCompleted { routing, .. } => {
                            state.routing = Some(RoutingSnapshot {
                                model_routing: routing.clone(),
                                tool_necessity: None,
                                tool_necessity_score: None,
                            });
                        }
                        PipelineEvent::ValidationPassed => {
                            // Available commands were set by the validate activity.
                        }
                        PipelineEvent::ResponseAssembled { response } => {
                            state.response = Some(response.clone());
                        }
                        PipelineEvent::Signal(Signal::Cancel { reason }) => {
                            cancel.cancel();
                            return Ok(Some(ReactorError::Cancelled {
                                reason: reason.clone(),
                            }));
                        }
                        _ => {}
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    return Err(ReactorError::ChannelClosed);
                }
            }
        }
        Ok(None)
    }

    /// Execute all queued commands sequentially.
    ///
    /// Returns Some(err) if execution should terminate, None on success.
    #[allow(clippy::too_many_arguments)]
    async fn execute_commands(
        &self,
        execution_id: &ExecutionId,
        commands: &mut Vec<weft_core::CommandInvocation>,
        state: &mut ExecutionState,
        request: &weft_core::WeftRequest,
        pipeline: &CompiledPipeline,
        event_tx: &mpsc::Sender<PipelineEvent>,
        event_rx: &mut mpsc::Receiver<PipelineEvent>,
        cancel: &CancellationToken,
        default_timeout_secs: u64,
    ) -> Result<Option<ReactorError>, ReactorError> {
        for (cmd_index, invocation) in commands.drain(..).enumerate() {
            if cancel.is_cancelled() {
                break;
            }

            // Idempotency key for this command.
            let cmd_key = format!(
                "{}:{}:{}:{}",
                execution_id, invocation.name, state.iteration, cmd_index
            );

            // Check idempotency — skip if already completed.
            if self
                .check_idempotency(execution_id, &cmd_key)
                .await?
                .is_some()
            {
                debug!(key = %cmd_key, "command idempotency hit: skipping");
                state.commands_executed += 1;
                continue;
            }

            // Run pre_tool_use hooks.
            for hook in &pipeline.loop_hooks.pre_tool_use {
                if cancel.is_cancelled() {
                    break;
                }
                let input = self.build_input(execution_id, state, request, hook, None);
                hook.activity
                    .execute(
                        execution_id,
                        input,
                        &self.services,
                        self.event_log.as_ref(),
                        event_tx.clone(),
                        cancel.clone(),
                    )
                    .await;
                let terminate = self
                    .drain_pre_post_loop(execution_id, event_rx, state, cancel)
                    .await?;
                if let Some(err) = terminate {
                    return Ok(Some(err));
                }
            }

            // Build command input with idempotency key.
            let mut cmd_input = self.build_input(
                execution_id,
                state,
                request,
                &pipeline.execute_command,
                Some(cmd_key),
            );
            // Inject the specific invocation into metadata so the activity knows which command to run.
            cmd_input.metadata = serde_json::to_value(&invocation).unwrap_or_default();

            let cmd_timeout = Duration::from_secs(
                pipeline
                    .execute_command
                    .timeout_secs
                    .unwrap_or(default_timeout_secs),
            );

            // Execute command with timeout.
            let cmd_activity = Arc::clone(&pipeline.execute_command.activity);
            let cmd_event_tx = event_tx.clone();
            let cmd_exec_id = execution_id.clone();
            let cmd_services = Arc::clone(&self.services);
            let cmd_event_log: Arc<dyn EventLog> = Arc::clone(&self.event_log);
            let cmd_cancel = cancel.clone();

            let cmd_handle = tokio::spawn(async move {
                cmd_activity
                    .execute(
                        &cmd_exec_id,
                        cmd_input,
                        &cmd_services,
                        cmd_event_log.as_ref(),
                        cmd_event_tx,
                        cmd_cancel,
                    )
                    .await;
            });

            // Drain events until the command completes.
            let cmd_deadline = tokio::time::Instant::now() + cmd_timeout;
            let mut cmd_completed = false;
            let mut cmd_failed = false;
            let mut cmd_failed_error = String::new();
            let mut cmd_failed_retryable = false;
            let cmd_name = invocation.name.clone();

            'cmd: loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        cmd_handle.abort();
                        break 'cmd;
                    }
                    event_opt = event_rx.recv() => {
                        match event_opt {
                            None => return Err(ReactorError::ChannelClosed),
                            Some(event) => {
                                self.record_event(execution_id, &event).await?;
                                match &event {
                                    PipelineEvent::CommandCompleted { name, result } if name == &cmd_name => {
                                        // Inject command result into messages.
                                        let result_msg = weft_core::WeftMessage {
                                            role: weft_core::Role::User,
                                            source: weft_core::Source::Client,
                                            model: None,
                                            content: vec![weft_core::ContentPart::Text(result.output.clone())],
                                            delta: false,
                                            message_index: state.messages.len() as u32,
                                        };
                                        state.messages.push(result_msg);
                                        state.commands_executed += 1;
                                        cmd_completed = true;
                                    }
                                    PipelineEvent::CommandFailed { name, error } if name == &cmd_name => {
                                        // Inject error into messages.
                                        let err_msg = weft_core::WeftMessage {
                                            role: weft_core::Role::User,
                                            source: weft_core::Source::Client,
                                            model: None,
                                            content: vec![weft_core::ContentPart::Text(format!("Command failed: {error}"))],
                                            delta: false,
                                            message_index: state.messages.len() as u32,
                                        };
                                        state.messages.push(err_msg);
                                        state.commands_executed += 1;
                                        cmd_completed = true;
                                    }
                                    PipelineEvent::ActivityFailed { name, error, retryable } if name == pipeline.execute_command.activity.name() => {
                                        cmd_failed = true;
                                        cmd_failed_error = error.clone();
                                        cmd_failed_retryable = *retryable;
                                    }
                                    PipelineEvent::ActivityCompleted { name, .. } if name == pipeline.execute_command.activity.name() => {
                                        cmd_completed = true;
                                    }
                                    _ => {}
                                }
                                if cmd_completed || cmd_failed {
                                    break 'cmd;
                                }
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(cmd_deadline) => {
                        cmd_handle.abort();
                        cmd_failed = true;
                        cmd_failed_retryable = true;
                        cmd_failed_error = format!("command timed out after {} secs", cmd_timeout.as_secs());
                        break 'cmd;
                    }
                }
            }

            // Handle command failure with retry.
            if cmd_failed {
                let policy = pipeline.execute_command.retry_policy.as_ref();
                if cmd_failed_retryable && should_retry(policy, 0, &state.budget, cancel) {
                    // Command retry is not yet implemented. The spec's "on ActivityFailed with
                    // retryable=true, apply retry policy" applies to generation activities.
                    // Command-level retry requires per-command attempt tracking and a retry loop
                    // equivalent to the generate dispatch loop, which is deferred to a future
                    // phase. For now, log a warning and propagate the failure. Commands that need
                    // retry should handle it internally (e.g., via idempotent re-submission).
                    warn!(
                        "command '{}' failed (retryable=true): {}; command retry not yet implemented",
                        cmd_name, cmd_failed_error
                    );
                }
                return Ok(Some(ReactorError::ActivityFailed(
                    crate::activity::ActivityError::Failed {
                        name: cmd_name,
                        reason: cmd_failed_error,
                    },
                )));
            }

            // Run post_tool_use hooks.
            for hook in &pipeline.loop_hooks.post_tool_use {
                if cancel.is_cancelled() {
                    break;
                }
                let input = self.build_input(execution_id, state, request, hook, None);
                hook.activity
                    .execute(
                        execution_id,
                        input,
                        &self.services,
                        self.event_log.as_ref(),
                        event_tx.clone(),
                        cancel.clone(),
                    )
                    .await;
                let terminate = self
                    .drain_pre_post_loop(execution_id, event_rx, state, cancel)
                    .await?;
                if let Some(err) = terminate {
                    return Ok(Some(err));
                }
            }
        }
        Ok(None)
    }

    /// Dispatch a PipelineEvent from the generation phase to update state.
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_generated_event(
        &self,
        _execution_id: &ExecutionId,
        event: &PipelineEvent,
        state: &mut ExecutionState,
        client_tx: &Option<mpsc::Sender<PipelineEvent>>,
        commands_queued: &mut Vec<weft_core::CommandInvocation>,
        generation_done: &mut bool,
        generation_refused: &mut bool,
    ) -> Result<(), ReactorError> {
        if let PipelineEvent::Generated(gen_ev) = event {
            if let GeneratedEvent::Content { .. } = gen_ev
                && let Some(client) = client_tx
            {
                let _ = client.send(event.clone()).await;
            }
            match gen_ev {
                GeneratedEvent::Content { part } => {
                    if let weft_core::ContentPart::Text(t) = part {
                        state.accumulated_text.push_str(t);
                    }
                }
                GeneratedEvent::CommandInvocation(inv) => {
                    commands_queued.push(inv.clone());
                }
                GeneratedEvent::Done => {
                    *generation_done = true;
                }
                GeneratedEvent::Refused { .. } => {
                    *generation_refused = true;
                }
                GeneratedEvent::Reasoning { .. } => {}
            }
        }
        Ok(())
    }

    /// Record a PipelineEvent to the EventLog.
    ///
    /// Derives event_type string from variant, serializes to JSON,
    /// and passes EVENT_SCHEMA_VERSION. Extracts idempotency_key from
    /// ActivityCompleted events if present.
    async fn record_event(
        &self,
        execution_id: &ExecutionId,
        event: &PipelineEvent,
    ) -> Result<u64, ReactorError> {
        let event_type = event.event_type_string();
        let payload = serde_json::to_value(event)?;
        let idempotency_key = match event {
            PipelineEvent::ActivityCompleted {
                idempotency_key: Some(key),
                ..
            } => Some(key.as_str()),
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
    async fn check_idempotency(
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
    fn extract_activity_events(&self, idempotency_key: &str, events: &[Event]) -> Vec<Event> {
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
    async fn finalize_failed(
        &self,
        execution_id: &ExecutionId,
        state: &mut ExecutionState,
        err: ReactorError,
    ) -> Result<(), ReactorError> {
        self.record_event(
            execution_id,
            &PipelineEvent::ExecutionFailed {
                error: err.to_string(),
                partial_text: Some(state.accumulated_text.clone()),
            },
        )
        .await?;
        self.event_log
            .update_execution_status(execution_id, ExecutionStatus::Failed)
            .await?;
        Ok(())
    }
}

// ── Retry helpers ─────────────────────────────────────────────────────────────

/// Whether the activity should be retried.
///
/// `attempt` is 0-indexed: 0 = just failed the initial attempt.
fn should_retry(
    policy: Option<&RetryPolicy>,
    attempt: u32,
    budget: &Budget,
    cancel: &CancellationToken,
) -> bool {
    let Some(policy) = policy else { return false };
    attempt < policy.max_retries
        && !matches!(budget.check(), BudgetCheck::Exhausted(_))
        && !cancel.is_cancelled()
}

/// Compute backoff duration in milliseconds with 0-25% jitter.
fn backoff_ms(policy: &RetryPolicy, attempt: u32) -> u64 {
    let base = policy.initial_backoff_ms as f64 * policy.backoff_multiplier.powi(attempt as i32);
    let capped = base.min(policy.max_backoff_ms as f64) as u64;
    // Add 0-25% jitter to prevent thundering herd.
    let jitter = rand::random::<u64>() % (capped / 4 + 1);
    capped + jitter
}

// ── Helper: empty response ────────────────────────────────────────────────────

fn empty_response(execution_id: &ExecutionId) -> weft_core::WeftResponse {
    weft_core::WeftResponse {
        id: execution_id.to_string(),
        model: "unknown".to_string(),
        messages: vec![],
        usage: weft_core::WeftUsage::default(),
        timing: weft_core::WeftTiming::default(),
    }
}

// ── Test hooks (unit test access to private free functions) ───────────────────

#[cfg(test)]
pub mod test_hooks {
    use super::*;

    /// Expose `should_retry` for unit testing.
    pub fn should_retry_pub(
        policy: Option<&RetryPolicy>,
        attempt: u32,
        budget: &Budget,
        cancel: &CancellationToken,
    ) -> bool {
        should_retry(policy, attempt, budget, cancel)
    }

    /// Expose `backoff_ms` for unit testing.
    pub fn backoff_ms_pub(policy: &RetryPolicy, attempt: u32) -> u64 {
        backoff_ms(policy, attempt)
    }

    /// Expose `Reactor::check_idempotency` for unit testing.
    ///
    /// Allows tests to call the idempotency check directly with a known
    /// execution_id rather than going through `execute()` which always creates
    /// a fresh id.
    pub async fn check_idempotency_pub(
        reactor: &Reactor,
        execution_id: &ExecutionId,
        key: &str,
    ) -> Result<Option<Vec<crate::event::Event>>, crate::error::ReactorError> {
        reactor.check_idempotency(execution_id, key).await
    }
}
