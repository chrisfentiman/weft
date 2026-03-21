//! Resource budget for pipeline executions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::event::BudgetSnapshot;
use crate::signal::BudgetUpdate;

/// Retry configuration for activities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Resource limits for an execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub max_depth: u32,
    pub current_depth: u32,
    pub max_generation_calls: u32,
    pub remaining_generation_calls: u32,
    pub max_iterations: u32,
    pub remaining_iterations: u32,
    pub deadline: DateTime<Utc>,
}

/// Result of a budget check.
#[derive(Debug, Clone)]
pub enum BudgetCheck {
    Ok,
    Warning(BudgetWarningInfo),
    Exhausted(BudgetExhaustedReason),
}

#[derive(Debug, Clone)]
pub struct BudgetWarningInfo {
    pub resource: String,
    pub remaining: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetExhaustedReason {
    GenerationCalls,
    Iterations,
    Depth,
    Deadline,
}

impl std::fmt::Display for BudgetExhaustedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenerationCalls => write!(f, "generation_calls"),
            Self::Iterations => write!(f, "iterations"),
            Self::Depth => write!(f, "depth"),
            Self::Deadline => write!(f, "deadline"),
        }
    }
}

impl Budget {
    pub fn new(
        max_generation_calls: u32,
        max_iterations: u32,
        max_depth: u32,
        deadline: DateTime<Utc>,
    ) -> Self {
        Self {
            max_depth,
            current_depth: 0,
            max_generation_calls,
            remaining_generation_calls: max_generation_calls,
            max_iterations,
            remaining_iterations: max_iterations,
            deadline,
        }
    }

    pub fn check(&self) -> BudgetCheck {
        if Utc::now() > self.deadline {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::Deadline);
        }
        if self.remaining_generation_calls == 0 {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::GenerationCalls);
        }
        if self.remaining_iterations == 0 {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::Iterations);
        }
        if self.current_depth >= self.max_depth {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::Depth);
        }
        if self.remaining_generation_calls == 1 {
            return BudgetCheck::Warning(BudgetWarningInfo {
                resource: "generation_calls".to_string(),
                remaining: 1,
            });
        }
        BudgetCheck::Ok
    }

    pub fn record_generation(&mut self) -> Result<(), BudgetExhaustedReason> {
        if self.remaining_generation_calls == 0 {
            return Err(BudgetExhaustedReason::GenerationCalls);
        }
        self.remaining_generation_calls -= 1;
        Ok(())
    }

    pub fn record_iteration(&mut self) -> Result<(), BudgetExhaustedReason> {
        if self.remaining_iterations == 0 {
            return Err(BudgetExhaustedReason::Iterations);
        }
        self.remaining_iterations -= 1;
        Ok(())
    }

    pub fn child_budget(&self) -> Result<Self, BudgetExhaustedReason> {
        let new_depth = self.current_depth + 1;
        if new_depth >= self.max_depth {
            return Err(BudgetExhaustedReason::Depth);
        }
        Ok(Self {
            max_depth: self.max_depth,
            current_depth: new_depth,
            max_generation_calls: self.remaining_generation_calls,
            remaining_generation_calls: self.remaining_generation_calls,
            max_iterations: self.remaining_iterations,
            remaining_iterations: self.remaining_iterations,
            deadline: self.deadline,
        })
    }

    pub fn apply_update(&mut self, update: BudgetUpdate) {
        if let Some(new_max) = update.max_generation_calls {
            let used = self
                .max_generation_calls
                .saturating_sub(self.remaining_generation_calls);
            self.max_generation_calls = new_max;
            self.remaining_generation_calls = new_max.saturating_sub(used);
        }
        if let Some(new_max) = update.max_iterations {
            let used = self
                .max_iterations
                .saturating_sub(self.remaining_iterations);
            self.max_iterations = new_max;
            self.remaining_iterations = new_max.saturating_sub(used);
        }
        if let Some(new_deadline) = update.deadline {
            self.deadline = new_deadline;
        }
    }

    pub fn deduct_child_usage(&mut self, child: &Budget) {
        let child_gen_used = child
            .max_generation_calls
            .saturating_sub(child.remaining_generation_calls);
        let child_iter_used = child
            .max_iterations
            .saturating_sub(child.remaining_iterations);
        self.remaining_generation_calls = self
            .remaining_generation_calls
            .saturating_sub(child_gen_used);
        self.remaining_iterations = self.remaining_iterations.saturating_sub(child_iter_used);
    }

    pub fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            max_generation_calls: self.max_generation_calls,
            remaining_generation_calls: self.remaining_generation_calls,
            max_iterations: self.max_iterations,
            remaining_iterations: self.remaining_iterations,
            max_depth: self.max_depth,
            current_depth: self.current_depth,
            deadline_epoch_ms: self.deadline.timestamp_millis(),
        }
    }
}
