//! Signal types for external control of running executions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// An external event injected into a running execution.
///
/// Signals are events on the channel. External systems push them
/// via the event channel sender, not a separate mechanism.
///
/// Signal delivery:
/// - **In-memory:** External callers hold an `mpsc::Sender<PipelineEvent>`
///   and push `PipelineEvent::Signal(signal)` onto the channel.
/// - **Persistent (Phase 6):** Postgres LISTEN/NOTIFY pushes signals
///   onto the channel via a background task.
///
/// Since signals are channel events, they are processed in the same
/// dispatch loop as all other events. No separate polling needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    /// Stop execution. Save partial results.
    Cancel { reason: String },

    /// Add messages to the execution context before the next generation call.
    InjectContext {
        messages: Vec<weft_core::WeftMessage>,
    },

    /// Adjust resource budget mid-execution.
    UpdateBudget { changes: BudgetUpdate },

    /// Override generation configuration for remaining calls.
    /// For LLMs, the config contains model selection. For other
    /// sources, it contains whatever the source needs.
    ForceGenerationConfig { config: serde_json::Value },

    /// Pause execution at the next dispatch. The Reactor stops
    /// processing events until a Resume signal arrives.
    Pause,

    /// Resume a paused execution.
    Resume,
}

impl Signal {
    /// Returns the signal type as a string for event logging.
    pub fn signal_type(&self) -> &'static str {
        match self {
            Signal::Cancel { .. } => "cancel",
            Signal::InjectContext { .. } => "inject_context",
            Signal::UpdateBudget { .. } => "update_budget",
            Signal::ForceGenerationConfig { .. } => "force_generation_config",
            Signal::Pause => "pause",
            Signal::Resume => "resume",
        }
    }
}

/// Budget adjustment fields. All fields are optional -- only present
/// fields are applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetUpdate {
    pub max_generation_calls: Option<u32>,
    pub max_iterations: Option<u32>,
    pub deadline: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_strings() {
        assert_eq!(
            Signal::Cancel {
                reason: "test".to_string()
            }
            .signal_type(),
            "cancel"
        );
        assert_eq!(
            Signal::InjectContext { messages: vec![] }.signal_type(),
            "inject_context"
        );
        assert_eq!(
            Signal::UpdateBudget {
                changes: BudgetUpdate {
                    max_generation_calls: None,
                    max_iterations: None,
                    deadline: None,
                }
            }
            .signal_type(),
            "update_budget"
        );
        assert_eq!(
            Signal::ForceGenerationConfig {
                config: serde_json::Value::Null
            }
            .signal_type(),
            "force_generation_config"
        );
        assert_eq!(Signal::Pause.signal_type(), "pause");
        assert_eq!(Signal::Resume.signal_type(), "resume");
    }

    #[test]
    fn test_signal_cancel_serde() {
        let sig = Signal::Cancel {
            reason: "user requested".to_string(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signal_type(), "cancel");
    }

    #[test]
    fn test_budget_update_partial() {
        let update = BudgetUpdate {
            max_generation_calls: Some(10),
            max_iterations: None,
            deadline: None,
        };
        let json = serde_json::to_string(&update).unwrap();
        let back: BudgetUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_generation_calls, Some(10));
        assert!(back.max_iterations.is_none());
    }
}
