//! EventLog trait and error type — re-exported from `weft_reactor_trait`.
//!
//! The canonical definitions live in `weft_reactor_trait`. This module re-exports
//! them so the existing `use weft_reactor::event_log::*` paths keep working.

pub use weft_reactor_trait::{EventLog, EventLogError};
