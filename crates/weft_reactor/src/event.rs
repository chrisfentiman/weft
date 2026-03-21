//! Pipeline event types — re-exported from `weft_reactor_trait`.
//!
//! The canonical definitions live in `weft_reactor_trait`. This module re-exports
//! them so the existing `use weft_reactor::event::*` paths keep working.

pub use weft_reactor_trait::{
    ActivityEvent, BudgetEvent, BudgetSnapshot, ChildEvent, CommandEvent, CommandFormat,
    ContextEvent, EVENT_SCHEMA_VERSION, Event, ExecutionEvent, GeneratedEvent, GenerationEvent,
    HookOutcome, MessageInjectionSource, PipelineEvent, SelectionEvent, SignalEvent,
};
