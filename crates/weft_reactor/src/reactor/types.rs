//! Internal types for the Reactor execution engine.
//!
//! This module defines the data structures used internally by the reactor
//! during execution. Types here are `pub(super)` -- visible to sibling
//! submodules within `reactor/` but not accessible outside the module.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::activity::{Activity, RoutingSnapshot};
use crate::budget::Budget;
use crate::config::{PipelineConfig, RetryPolicy};
use crate::event::{CommandFormat, DegradationNotice, PipelineEvent};
use crate::execution::ExecutionId;
use weft_core::ResolvedConfig;

// ── Public types ──────────────────────────────────────────────────────────────

/// Collects the parameters for `Reactor::execute` into a single owned struct.
///
/// Replaces the previous 7-parameter signature. `cancel` is kept separate
/// because it is a borrowed reference — embedding it would require a lifetime
/// parameter on the struct itself.
pub struct ExecutionContext {
    pub request: weft_core::WeftRequest,
    pub tenant_id: crate::execution::TenantId,
    pub request_id: crate::execution::RequestId,
    pub parent_id: Option<ExecutionId>,
    pub parent_budget: Option<Budget>,
    pub client_tx: Option<mpsc::Sender<PipelineEvent>>,
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

// ── Private/internal types ─────────────────────────────────────────────────────

/// A pipeline with all activity references resolved.
pub(super) struct CompiledPipeline {
    #[allow(dead_code)]
    pub(super) config: PipelineConfig,
    pub(super) pre_loop: Vec<ResolvedActivity>,
    pub(super) post_loop: Vec<ResolvedActivity>,
    pub(super) generate: ResolvedActivity,
    pub(super) execute_command: ResolvedActivity,
    pub(super) loop_hooks: CompiledLoopHooks,
}

/// An activity reference resolved from the registry, carrying its
/// runtime config (retry policy, timeout, heartbeat interval).
pub(super) struct ResolvedActivity {
    pub(super) activity: Arc<dyn Activity>,
    pub(super) retry_policy: Option<RetryPolicy>,
    pub(super) timeout_secs: Option<u64>,
    pub(super) heartbeat_interval_secs: Option<u64>,
}

pub(super) struct CompiledLoopHooks {
    pub(super) pre_generate: Vec<ResolvedActivity>,
    pub(super) pre_response: Vec<ResolvedActivity>,
    pub(super) pre_tool_use: Vec<ResolvedActivity>,
    pub(super) post_tool_use: Vec<ResolvedActivity>,
}

/// Internal state maintained by the Reactor during execution.
/// Not passed to activities — they receive ActivityInput snapshots.
pub(super) struct ExecutionState {
    /// Working message list.
    pub(super) messages: Vec<weft_core::WeftMessage>,
    /// Current budget state.
    pub(super) budget: Budget,
    /// Routing result, set from RouteCompleted events.
    /// Maintained for backward compatibility: populated by the ModelSelected
    /// drain arm so GenerateActivity continues to receive routing_result.
    pub(super) routing: Option<RoutingSnapshot>,
    /// Generation config override from ForceGenerationConfig signal.
    pub(super) generation_config_override: Option<serde_json::Value>,
    /// Accumulated text content across all generation calls.
    pub(super) accumulated_text: String,
    /// Available commands, populated by ValidateActivity.
    pub(super) available_commands: Vec<weft_core::CommandStub>,
    /// The assembled response, set from ResponseAssembled events.
    pub(super) response: Option<weft_core::WeftResponse>,
    /// Count of commands executed across all iterations.
    pub(super) commands_executed: u32,
    /// Execution start time for duration tracking.
    pub(super) start_time: Instant,
    /// Current dispatch loop iteration (0-indexed).
    pub(super) iteration: u32,
    /// Last event timestamp per spawned activity, for heartbeat tracking.
    pub(super) last_activity_event: HashMap<String, Instant>,
    /// Current retry attempt for the generate activity (0 = initial attempt).
    pub(super) generate_retry_attempt: u32,
    /// Accumulated token usage across all generation calls.
    pub(super) accumulated_usage: weft_core::WeftUsage,

    // ── Degradation tracking (Phase 2+) ───────────────────────────────
    /// Degradation notices accumulated during execution.
    /// Populated by run_pre_loop and run_post_loop when non-critical activities fail.
    pub(super) degradations: Vec<DegradationNotice>,
    /// Per-request resolved configuration snapshot. Used by degradation fallback
    /// logic (e.g., model_selection fallback reads config.default_model).
    pub(super) config: Arc<ResolvedConfig>,

    // ── Fields set by new pre-loop activities (Phase 1+) ─────────────
    /// Selected model routing name. Set by ModelSelectionActivity.
    pub(super) selected_model: Option<String>,
    /// Selected model API identifier. Set by ProviderResolutionActivity.
    pub(super) selected_model_id: Option<String>,
    /// Provider name for the selected model. Set by ProviderResolutionActivity.
    pub(super) selected_provider: Option<String>,
    /// Capabilities of the selected model, as strings. Set by ProviderResolutionActivity.
    /// Converted from `HashSet<Capability>` via `cap.as_str().to_string()`.
    pub(super) model_capabilities: Vec<String>,
    /// Max tokens for the selected model. Set by ProviderResolutionActivity.
    pub(super) model_max_tokens: Option<u32>,
    /// Commands selected by semantic routing. Set by CommandSelectionActivity.
    pub(super) selected_commands: Vec<weft_core::CommandStub>,
    /// How commands are formatted for the provider. Set by CommandFormattingActivity.
    pub(super) command_format: Option<CommandFormat>,
    /// Sampling max_tokens after clamping. Set by SamplingAdjustmentActivity.
    pub(super) sampling_max_tokens: Option<u32>,
    /// Sampling temperature. Set by SamplingAdjustmentActivity.
    pub(super) sampling_temperature: Option<f32>,
    /// Sampling top_p. Set by SamplingAdjustmentActivity.
    pub(super) sampling_top_p: Option<f32>,
}

impl ExecutionState {
    pub(super) fn new(budget: Budget, config: Arc<ResolvedConfig>) -> Self {
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
            degradations: Vec::new(),
            config,
            selected_model: None,
            selected_model_id: None,
            selected_provider: None,
            model_capabilities: Vec::new(),
            model_max_tokens: None,
            selected_commands: Vec::new(),
            command_format: None,
            sampling_max_tokens: None,
            sampling_temperature: None,
            sampling_top_p: None,
        }
    }
}

/// Bundles all borrows needed by the loop phases (pre-loop, dispatch, post-loop).
///
/// All fields are borrowed from the `execute` scope. The lifetime `'a` ties
/// them together so each phase method takes a single parameter instead of many.
pub(super) struct LoopContext<'a> {
    pub(super) execution_id: &'a ExecutionId,
    pub(super) state: &'a mut ExecutionState,
    pub(super) request: &'a weft_core::WeftRequest,
    pub(super) pipeline: &'a CompiledPipeline,
    pub(super) event_tx: &'a mpsc::Sender<PipelineEvent>,
    pub(super) event_rx: &'a mut mpsc::Receiver<PipelineEvent>,
    pub(super) cancel: &'a CancellationToken,
}

/// Bundles all borrows needed by `execute_commands` into a single struct.
///
/// All fields are borrowed from the enclosing `execute` dispatch loop scope.
/// The lifetime `'a` ties them together so the borrow checker can reason
/// about the lifetimes without threading 9 separate parameters.
pub(super) struct CommandContext<'a> {
    pub(super) execution_id: &'a ExecutionId,
    pub(super) state: &'a mut ExecutionState,
    pub(super) request: &'a weft_core::WeftRequest,
    pub(super) pipeline: &'a CompiledPipeline,
    pub(super) event_tx: &'a mpsc::Sender<PipelineEvent>,
    pub(super) event_rx: &'a mut mpsc::Receiver<PipelineEvent>,
    pub(super) cancel: &'a CancellationToken,
    pub(super) default_timeout_secs: u64,
}
