//! Top-level reactor error type.

use crate::activity::ActivityError;
use crate::budget::BudgetExhaustedReason;
use crate::event_log::EventLogError;
use crate::registry::RegistryError;

/// Top-level error type for the Reactor and its components.
#[derive(Debug, thiserror::Error)]
pub enum ReactorError {
    /// An activity pushed `ActivityFailed` and the Reactor could not continue.
    #[error("activity failed: {0}")]
    ActivityFailed(#[from] ActivityError),

    /// The EventLog returned an error. Infrastructure failure.
    #[error("event log error: {0}")]
    EventLog(#[from] EventLogError),

    /// The named pipeline was not found in the reactor config.
    #[error("pipeline not found: {0}")]
    PipelineNotFound(String),

    /// An activity name referenced in a pipeline config was not in the registry.
    #[error("activity not found in registry: {0}")]
    ActivityNotFound(String),

    /// Budget was exhausted during execution.
    #[error("budget exhausted: {0}")]
    BudgetExhausted(String),

    /// A hook blocked the request at a blocking lifecycle point.
    #[error("hook blocked: {hook_name}: {reason}")]
    HookBlocked { hook_name: String, reason: String },

    /// Execution was cancelled via Signal::Cancel or CancellationToken.
    #[error("execution cancelled: {reason}")]
    Cancelled { reason: String },

    /// Pipeline or reactor configuration is invalid.
    #[error("configuration error: {0}")]
    Config(String),

    /// Registry lookup error during compilation.
    #[error("registry error: {0}")]
    Registry(#[from] RegistryError),

    /// JSON serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The event channel receiver was dropped unexpectedly.
    /// Indicates a logic bug or panic in the runtime.
    #[error("channel error: receiver dropped")]
    ChannelClosed,
}

impl From<serde_json::Error> for ReactorError {
    fn from(e: serde_json::Error) -> Self {
        ReactorError::Serialization(e.to_string())
    }
}

impl From<BudgetExhaustedReason> for ReactorError {
    fn from(reason: BudgetExhaustedReason) -> Self {
        ReactorError::BudgetExhausted(reason.to_string())
    }
}
