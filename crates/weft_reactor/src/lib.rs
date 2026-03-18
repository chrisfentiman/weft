//! `weft_reactor` — Reactive pipeline primitives for the Weft gateway.
//!
//! This crate owns the 8 primitives of the reactive pipeline:
//! 1. **Execution** -- identity, lifecycle, parent-child relationships
//! 2. **PipelineEvent** -- the unified event type on the channel
//! 3. **Signal** -- external control, delivered as events on the channel
//! 4. **Activity** -- event-producing unit of work (Phase 2+)
//! 5. **EventLog** -- pluggable durable store trait
//! 6. **Reactor** -- dispatch loop (Phase 4+)
//! 7. **Budget** -- resource limits
//! 8. **Services** -- shared infrastructure (Phase 5+)
//!
//! EventLog implementations live in separate crates:
//! - `weft_eventlog_memory` -- for tests and local dev
//! - `weft_eventlog_postgres` -- for production (Phase 6)

pub mod budget;
pub mod error;
pub mod event;
pub mod event_log;
pub mod execution;
pub mod signal;

// Re-exports for convenience
pub use budget::{Budget, BudgetCheck, BudgetExhaustedReason, BudgetWarningInfo, RetryPolicy};
pub use error::ReactorError;
pub use event::{BudgetSnapshot, EVENT_SCHEMA_VERSION, Event, GeneratedEvent, PipelineEvent};
pub use event_log::{EventLog, EventLogError};
pub use execution::{Execution, ExecutionId, ExecutionStatus, RequestId, TenantId};
pub use signal::{BudgetUpdate, Signal};
