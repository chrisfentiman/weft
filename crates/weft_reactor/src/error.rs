//! Top-level reactor error type.

use crate::event_log::EventLogError;

/// Top-level error type for the Reactor and its components.
#[derive(Debug, thiserror::Error)]
pub enum ReactorError {
    #[error("event log error: {0}")]
    EventLog(#[from] EventLogError),

    #[error("budget exhausted: {reason}")]
    BudgetExhausted { reason: String },

    #[error("execution cancelled: {reason}")]
    Cancelled { reason: String },

    #[error("activity error in '{name}': {reason}")]
    Activity { name: String, reason: String },

    #[error("pipeline configuration error: {0}")]
    Config(String),

    #[error("serialization error: {0}")]
    Serialization(String),
}
