//! `weft_reactor` — Reactive pipeline primitives for the Weft gateway.
//!
//! This crate owns the 8 primitives of the reactive pipeline:
//! 1. **Execution** -- identity, lifecycle, parent-child relationships
//! 2. **PipelineEvent** -- the unified event type on the channel
//! 3. **Signal** -- external control, delivered as events on the channel
//! 4. **Activity** -- event-producing unit of work
//! 5. **EventLog** -- pluggable durable store trait
//! 6. **Reactor** -- dispatch loop (Phase 4+)
//! 7. **Budget** -- resource limits
//! 8. **Services** -- shared infrastructure
//!
//! EventLog implementations live in separate crates:
//! - `weft_eventlog_memory` -- for tests and local dev
//! - `weft_eventlog_postgres` -- for production (Phase 6)

pub mod activities;
pub mod activity;
pub mod budget;
pub mod config;
pub mod error;
pub mod event;
pub mod event_log;
pub mod execution;
pub mod reactor;
pub mod registry;
pub mod services;
pub mod signal;

#[cfg(test)]
pub mod test_support;

#[cfg(test)]
mod reactor_tests;

// Re-exports for convenience
pub use activity::{Activity, ActivityError, ActivityInput, RoutingSnapshot};
pub use budget::{Budget, BudgetCheck, BudgetExhaustedReason, BudgetWarningInfo};
pub use config::{
    ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig, RetryPolicy,
};
pub use error::ReactorError;
pub use event::{BudgetSnapshot, EVENT_SCHEMA_VERSION, Event, GeneratedEvent, PipelineEvent};
pub use event_log::{EventLog, EventLogError};
pub use execution::{Execution, ExecutionId, ExecutionStatus, RequestId, TenantId};
pub use reactor::{BudgetUsage, ExecutionResult, Reactor};
pub use registry::{ActivityRegistry, RegistryError};
pub use services::{ReactorHandle, Services};
pub use signal::{BudgetUpdate, Signal};
