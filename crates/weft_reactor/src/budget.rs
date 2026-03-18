//! Resource budget for pipeline executions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::event::BudgetSnapshot;
use crate::signal::BudgetUpdate;

/// Resource limits for an execution.
///
/// Budget is checked by the Reactor after processing each event.
/// When limits approach, the Budget pushes warning/exhaustion events
/// onto the channel via the `check_and_emit` method.
///
/// Budget is NOT shared across concurrent executions via Arc/Atomic.
/// Each execution owns its Budget. Child executions get a derived
/// budget via `child_budget()`. The parent decrements its own budget
/// by whatever the child consumed after the child completes.
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
    /// All limits are within bounds.
    Ok,
    /// A resource is running low. Contains warning info to emit.
    Warning(BudgetWarningInfo),
    /// A resource is exhausted. Execution must stop.
    Exhausted(BudgetExhaustedReason),
}

/// Information about a budget warning.
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
    /// Create a new budget with the given limits.
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

    /// Check budget state. Check order: deadline > generation_calls > iterations > depth.
    /// Returns Warning when remaining_generation_calls == 1.
    pub fn check(&self) -> BudgetCheck {
        // Deadline check first
        if Utc::now() > self.deadline {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::Deadline);
        }

        // Generation calls exhausted
        if self.remaining_generation_calls == 0 {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::GenerationCalls);
        }

        // Iterations exhausted
        if self.remaining_iterations == 0 {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::Iterations);
        }

        // Depth exhausted
        if self.current_depth >= self.max_depth {
            return BudgetCheck::Exhausted(BudgetExhaustedReason::Depth);
        }

        // Warn when only 1 generation call remains
        if self.remaining_generation_calls == 1 {
            return BudgetCheck::Warning(BudgetWarningInfo {
                resource: "generation_calls".to_string(),
                remaining: 1,
            });
        }

        BudgetCheck::Ok
    }

    /// Decrements remaining_generation_calls. Err if already 0.
    pub fn record_generation(&mut self) -> Result<(), BudgetExhaustedReason> {
        if self.remaining_generation_calls == 0 {
            return Err(BudgetExhaustedReason::GenerationCalls);
        }
        self.remaining_generation_calls -= 1;
        Ok(())
    }

    /// Decrements remaining_iterations. Err if already 0.
    pub fn record_iteration(&mut self) -> Result<(), BudgetExhaustedReason> {
        if self.remaining_iterations == 0 {
            return Err(BudgetExhaustedReason::Iterations);
        }
        self.remaining_iterations -= 1;
        Ok(())
    }

    /// Child inherits parent's remaining resources, depth + 1.
    /// Err(Depth) if current_depth + 1 >= max_depth.
    pub fn child_budget(&self) -> Result<Self, BudgetExhaustedReason> {
        let new_depth = self.current_depth + 1;
        if new_depth >= self.max_depth {
            return Err(BudgetExhaustedReason::Depth);
        }
        Ok(Self {
            max_depth: self.max_depth,
            current_depth: new_depth,
            max_generation_calls: self.max_generation_calls,
            remaining_generation_calls: self.remaining_generation_calls,
            max_iterations: self.max_iterations,
            remaining_iterations: self.remaining_iterations,
            deadline: self.deadline,
        })
    }

    /// Apply update: adjusts remaining proportionally to max change.
    /// Only present fields in BudgetUpdate are applied.
    ///
    /// When max is increased, remaining increases by the same delta.
    /// When max is decreased, remaining is clamped to the new max.
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

    /// Deduct child's consumption from parent. Uses saturating_sub.
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

    /// Snapshot for event payloads.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn future_deadline() -> DateTime<Utc> {
        Utc::now() + chrono::Duration::hours(1)
    }

    fn past_deadline() -> DateTime<Utc> {
        Utc::now() - chrono::Duration::seconds(1)
    }

    #[test]
    fn test_budget_new() {
        let b = Budget::new(10, 5, 3, future_deadline());
        assert_eq!(b.max_generation_calls, 10);
        assert_eq!(b.remaining_generation_calls, 10);
        assert_eq!(b.max_iterations, 5);
        assert_eq!(b.remaining_iterations, 5);
        assert_eq!(b.max_depth, 3);
        assert_eq!(b.current_depth, 0);
    }

    #[test]
    fn test_check_ok() {
        let b = Budget::new(10, 5, 3, future_deadline());
        assert!(matches!(b.check(), BudgetCheck::Ok));
    }

    #[test]
    fn test_check_deadline_exhausted() {
        let b = Budget::new(10, 5, 3, past_deadline());
        assert!(matches!(
            b.check(),
            BudgetCheck::Exhausted(BudgetExhaustedReason::Deadline)
        ));
    }

    #[test]
    fn test_check_generation_calls_exhausted() {
        let mut b = Budget::new(2, 5, 3, future_deadline());
        b.remaining_generation_calls = 0;
        assert!(matches!(
            b.check(),
            BudgetCheck::Exhausted(BudgetExhaustedReason::GenerationCalls)
        ));
    }

    #[test]
    fn test_check_iterations_exhausted() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.remaining_iterations = 0;
        assert!(matches!(
            b.check(),
            BudgetCheck::Exhausted(BudgetExhaustedReason::Iterations)
        ));
    }

    #[test]
    fn test_check_depth_exhausted() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.current_depth = 3; // >= max_depth
        assert!(matches!(
            b.check(),
            BudgetCheck::Exhausted(BudgetExhaustedReason::Depth)
        ));
    }

    #[test]
    fn test_check_warning_one_generation_remaining() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.remaining_generation_calls = 1;
        match b.check() {
            BudgetCheck::Warning(info) => {
                assert_eq!(info.resource, "generation_calls");
                assert_eq!(info.remaining, 1);
            }
            other => panic!("expected Warning, got {:?}", other),
        }
    }

    #[test]
    fn test_check_deadline_takes_priority_over_exhausted() {
        // Deadline should come first in check order
        let mut b = Budget::new(10, 5, 3, past_deadline());
        b.remaining_generation_calls = 0;
        assert!(matches!(
            b.check(),
            BudgetCheck::Exhausted(BudgetExhaustedReason::Deadline)
        ));
    }

    #[test]
    fn test_record_generation_decrements() {
        let mut b = Budget::new(5, 5, 3, future_deadline());
        b.record_generation().unwrap();
        assert_eq!(b.remaining_generation_calls, 4);
    }

    #[test]
    fn test_record_generation_errors_at_zero() {
        let mut b = Budget::new(1, 5, 3, future_deadline());
        b.record_generation().unwrap();
        let err = b.record_generation().unwrap_err();
        assert_eq!(err, BudgetExhaustedReason::GenerationCalls);
    }

    #[test]
    fn test_record_iteration_decrements() {
        let mut b = Budget::new(5, 5, 3, future_deadline());
        b.record_iteration().unwrap();
        assert_eq!(b.remaining_iterations, 4);
    }

    #[test]
    fn test_record_iteration_errors_at_zero() {
        let mut b = Budget::new(5, 1, 3, future_deadline());
        b.record_iteration().unwrap();
        let err = b.record_iteration().unwrap_err();
        assert_eq!(err, BudgetExhaustedReason::Iterations);
    }

    #[test]
    fn test_child_budget_increments_depth() {
        let b = Budget::new(10, 5, 3, future_deadline());
        let child = b.child_budget().unwrap();
        assert_eq!(child.current_depth, 1);
        assert_eq!(child.remaining_generation_calls, 10);
    }

    #[test]
    fn test_child_budget_fails_at_max_depth() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.current_depth = 2; // next child would be depth 3, which >= max_depth 3
        let err = b.child_budget().unwrap_err();
        assert_eq!(err, BudgetExhaustedReason::Depth);
    }

    #[test]
    fn test_child_budget_inherits_remaining() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.remaining_generation_calls = 7;
        b.remaining_iterations = 3;
        let child = b.child_budget().unwrap();
        assert_eq!(child.remaining_generation_calls, 7);
        assert_eq!(child.remaining_iterations, 3);
    }

    #[test]
    fn test_deduct_child_usage() {
        let mut parent = Budget::new(10, 5, 3, future_deadline());
        // Child started with parent's resources and used 3 gen calls and 2 iterations
        let mut child = parent.child_budget().unwrap();
        child.remaining_generation_calls = 7; // used 3
        child.remaining_iterations = 3; // used 2

        parent.deduct_child_usage(&child);
        assert_eq!(parent.remaining_generation_calls, 7); // 10 - 3
        assert_eq!(parent.remaining_iterations, 3); // 5 - 2
    }

    #[test]
    fn test_deduct_child_usage_saturating() {
        let mut parent = Budget::new(10, 5, 3, future_deadline());
        parent.remaining_generation_calls = 2;

        // Child claims it used more than parent has remaining
        let mut child = parent.child_budget().unwrap();
        child.remaining_generation_calls = 0; // used all 10

        parent.deduct_child_usage(&child);
        // saturating_sub prevents underflow
        assert_eq!(parent.remaining_generation_calls, 0);
    }

    #[test]
    fn test_apply_update_increases_max() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.remaining_generation_calls = 7; // used 3

        let update = BudgetUpdate {
            max_generation_calls: Some(15),
            max_iterations: None,
            deadline: None,
        };
        b.apply_update(update);
        assert_eq!(b.max_generation_calls, 15);
        // used=3, new_max=15, remaining=12
        assert_eq!(b.remaining_generation_calls, 12);
    }

    #[test]
    fn test_apply_update_decreases_max_clamps_remaining() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        b.remaining_generation_calls = 8; // used 2

        let update = BudgetUpdate {
            max_generation_calls: Some(3), // reduce max to 3, used=2, remaining=1
            max_iterations: None,
            deadline: None,
        };
        b.apply_update(update);
        assert_eq!(b.max_generation_calls, 3);
        assert_eq!(b.remaining_generation_calls, 1);
    }

    #[test]
    fn test_apply_update_deadline() {
        let mut b = Budget::new(10, 5, 3, future_deadline());
        let new_deadline = Utc::now() + chrono::Duration::hours(2);
        let update = BudgetUpdate {
            max_generation_calls: None,
            max_iterations: None,
            deadline: Some(new_deadline),
        };
        b.apply_update(update);
        assert_eq!(b.deadline, new_deadline);
    }

    #[test]
    fn test_snapshot() {
        let b = Budget::new(10, 5, 3, future_deadline());
        let snap = b.snapshot();
        assert_eq!(snap.max_generation_calls, 10);
        assert_eq!(snap.remaining_generation_calls, 10);
        assert_eq!(snap.max_depth, 3);
        assert_eq!(snap.current_depth, 0);
    }

    #[test]
    fn test_budget_exhausted_reason_display() {
        assert_eq!(
            BudgetExhaustedReason::GenerationCalls.to_string(),
            "generation_calls"
        );
        assert_eq!(BudgetExhaustedReason::Iterations.to_string(), "iterations");
        assert_eq!(BudgetExhaustedReason::Depth.to_string(), "depth");
        assert_eq!(BudgetExhaustedReason::Deadline.to_string(), "deadline");
    }

    #[test]
    fn test_budget_serde_round_trip() {
        let b = Budget::new(10, 5, 3, future_deadline());
        let json = serde_json::to_string(&b).unwrap();
        let back: Budget = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_generation_calls, 10);
        assert_eq!(back.max_depth, 3);
    }
}
