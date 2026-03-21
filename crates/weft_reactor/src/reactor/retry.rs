//! Retry helpers for the Reactor execution engine.
//!
//! `should_retry` and `backoff_ms` are free functions (not methods on Reactor)
//! because they operate only on their parameters with no need for `&self`.

use tokio_util::sync::CancellationToken;

use crate::budget::{Budget, BudgetCheck};
use crate::config::RetryPolicy;

/// Whether the activity should be retried.
///
/// `attempt` is 0-indexed: 0 = just failed the initial attempt.
pub(super) fn should_retry(
    policy: Option<&RetryPolicy>,
    attempt: u32,
    budget: &Budget,
    cancel: &CancellationToken,
) -> bool {
    let Some(policy) = policy else { return false };
    attempt < policy.max_retries
        && !matches!(budget.check(), BudgetCheck::Exhausted(_))
        && !cancel.is_cancelled()
}

/// Compute backoff duration in milliseconds with 0-25% jitter.
pub(super) fn backoff_ms(policy: &RetryPolicy, attempt: u32) -> u64 {
    let base = policy.initial_backoff_ms as f64 * policy.backoff_multiplier.powi(attempt as i32);
    let capped = base.min(policy.max_backoff_ms as f64) as u64;
    // Add 0-25% jitter to prevent thundering herd.
    let jitter = rand::random::<u64>() % (capped / 4 + 1);
    capped + jitter
}
