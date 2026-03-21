//! Signal types for external control of running executions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// An external event injected into a running execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    Cancel {
        reason: String,
    },
    InjectContext {
        messages: Vec<weft_core::WeftMessage>,
    },
    UpdateBudget {
        changes: BudgetUpdate,
    },
    ForceGenerationConfig {
        config: serde_json::Value,
    },
    Pause,
    Resume,
}

impl Signal {
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

/// Budget adjustment fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetUpdate {
    pub max_generation_calls: Option<u32>,
    pub max_iterations: Option<u32>,
    pub deadline: Option<DateTime<Utc>>,
}
