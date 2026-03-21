//! `weft_reactor_trait` — Reactor contract: event types, traits, and domain primitives.
//!
//! This crate defines the contract between the reactor engine and its consumers:
//! activities, event log implementations, and the binary crate. It contains no
//! implementations — only trait definitions and the types those traits operate on.
//!
//! # What lives here
//!
//! - **`PipelineEvent`** — grouped 10-outer / 39-leaf event enum (adjacently-tagged serde)
//! - **`EventLog`** trait — durable event store interface
//! - **`Activity`** trait — event-producing unit of work
//! - **`ServiceLocator`** trait — infrastructure abstraction for activities
//! - **`ChildSpawner`** trait — child execution spawning interface
//! - **`Budget`**, `BudgetCheck`, `RetryPolicy` — resource limits
//! - **`Signal`**, `BudgetUpdate` — external control signals
//! - **`ExecutionId`**, `TenantId`, `RequestId`, `Execution`, `ExecutionStatus` — identity
//!
//! # Who consumes this crate
//!
//! - `weft_reactor` — concrete implementation; re-exports everything here
//! - `weft_eventlog_memory` and `weft_eventlog_postgres` — EventLog implementations
//! - `weft_activities` (Phase 5) — activity implementations
//!
//! # Why it exists
//!
//! Before this crate, `weft_reactor` mixed contracts and implementations. This
//! split allows event log crates and future activity crates to depend on the
//! contract without pulling in the entire reactor implementation.

pub mod activity;
pub mod budget;
pub mod event;
pub mod event_log;
pub mod execution;
pub mod service;
pub mod signal;

// ── Public re-exports ──────────────────────────────────────────────────────

pub use activity::{Activity, ActivityError, ActivityInput, RoutingSnapshot, SemanticSelection};
pub use budget::{Budget, BudgetCheck, BudgetExhaustedReason, BudgetWarningInfo, RetryPolicy};
pub use event::{
    ActivityEvent, BudgetEvent, BudgetSnapshot, ChildEvent, CommandEvent, CommandFormat,
    ContextEvent, EVENT_SCHEMA_VERSION, Event, ExecutionEvent, GeneratedEvent, GenerationEvent,
    HookOutcome, MessageInjectionSource, PipelineEvent, SelectionEvent, SignalEvent,
};
pub use event_log::{EventLog, EventLogError};
pub use execution::{Execution, ExecutionId, ExecutionStatus, RequestId, TenantId};
pub use service::{ChildSpawner, ServiceLocator};
pub use signal::{BudgetUpdate, Signal};
