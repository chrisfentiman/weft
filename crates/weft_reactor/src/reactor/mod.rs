//! Reactor: dispatch loop consuming events from a channel.
//!
//! The Reactor is the execution engine for the Weft pipeline. It is NOT a
//! step-by-step executor. It is an event dispatch loop:
//!
//! 1. Run pre-loop activities sequentially (validate, model_selection, command_selection,
//!    provider_resolution, system_prompt_assembly, command_formatting, sampling_adjustment).
//! 2. Enter the dispatch loop:
//!    a. Spawn the generate activity as an event producer.
//!    b. Receive events from the channel via `tokio::select!`.
//!    c. Dispatch by variant (Generated, CommandInvocation, Signal, etc.).
//!    d. After commands complete, loop back to generation.
//! 3. Run post-loop activities sequentially (assemble_response).
//!
//! # Module layout
//!
//! - `types`    — all internal structs (ExecutionState, CompiledPipeline, etc.)
//! - `compile`  — Reactor::new and compile_pipeline
//! - `execute`  — Reactor::execute (setup + phase routing)
//! - `pre_post` — pre-loop and post-loop activity orchestration
//! - `dispatch` — the 'dispatch + 'generate select! loops
//! - `commands` — execute_commands and command-level dispatch
//! - `drain`    — drain_pre_post_loop
//! - `helpers`  — build_input, record_event, check_idempotency, finalize_failed
//! - `retry`    — should_retry, backoff_ms

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::BudgetConfig;
use crate::event_log::EventLog;
use crate::registry::ActivityRegistry;
use crate::services::Services;

use self::types::CompiledPipeline;

mod commands;
mod compile;
mod dispatch;
mod drain;
mod execute;
mod generate;
mod helpers;
mod pre_post;
mod retry;
mod types;

pub use types::{BudgetUsage, ExecutionContext, ExecutionResult};

#[cfg(any(test, feature = "test-support"))]
pub mod test_hooks;

/// The execution engine. A dispatch loop consuming events from a channel.
pub struct Reactor {
    services: Arc<Services>,
    event_log: Arc<dyn EventLog>,
    #[allow(dead_code)]
    registry: Arc<ActivityRegistry>,
    /// Compiled pipelines: activity references resolved to Arc<dyn Activity>.
    pipelines: HashMap<String, CompiledPipeline>,
    /// Default budget settings from ReactorConfig.
    budget_defaults: BudgetConfig,
}

impl std::fmt::Debug for Reactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reactor")
            .field("pipelines", &self.pipelines.keys().collect::<Vec<_>>())
            .finish()
    }
}
