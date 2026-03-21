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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_budget() -> Budget {
        Budget::new(10, 5, 3, chrono::Utc::now() + chrono::Duration::hours(1))
    }

    fn retry_policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy {
            max_retries,
            initial_backoff_ms: 100,
            max_backoff_ms: 1000,
            backoff_multiplier: 2.0,
        }
    }

    #[test]
    fn should_retry_with_policy_under_max() {
        let policy = retry_policy(3);
        let budget = test_budget();
        let cancel = CancellationToken::new();
        // attempt=0 < max_retries=3, budget ok, not cancelled → should retry
        assert!(should_retry(Some(&policy), 0, &budget, &cancel));
        assert!(should_retry(Some(&policy), 1, &budget, &cancel));
        assert!(should_retry(Some(&policy), 2, &budget, &cancel));
    }

    #[test]
    fn should_retry_exceeds_max_attempts() {
        let policy = retry_policy(2);
        let budget = test_budget();
        let cancel = CancellationToken::new();
        // attempt=2 is NOT < max_retries=2 → should not retry
        assert!(!should_retry(Some(&policy), 2, &budget, &cancel));
        assert!(!should_retry(Some(&policy), 3, &budget, &cancel));
    }

    #[test]
    fn should_retry_no_policy() {
        let budget = test_budget();
        let cancel = CancellationToken::new();
        assert!(!should_retry(None, 0, &budget, &cancel));
    }

    #[test]
    fn should_retry_false_when_cancelled() {
        let policy = retry_policy(5);
        let budget = test_budget();
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(!should_retry(Some(&policy), 0, &budget, &cancel));
    }

    #[test]
    fn backoff_ms_doubles_with_cap() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff_ms: 1000,
            max_backoff_ms: 5000,
            backoff_multiplier: 2.0,
        };
        // attempt=5: base = 1000 * 2^5 = 32000, capped to 5000
        let ms = backoff_ms(&policy, 5);
        assert!(
            ms >= 5000,
            "backoff should be at least max_backoff_ms (before jitter)"
        );
        // max jitter = 25% of capped = 1250, so max total = 6250
        assert!(
            ms < 5000 + 5000 / 4 + 2,
            "backoff should not exceed cap + 25% jitter; got {ms}"
        );
    }

    #[test]
    fn backoff_ms_grows_exponentially_below_cap() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff_ms: 10,
            max_backoff_ms: 100_000,
            backoff_multiplier: 2.0,
        };
        // Without cap interference, attempt=1 should be ~20ms, attempt=2 ~40ms
        let ms0 = backoff_ms(&policy, 0);
        let ms1 = backoff_ms(&policy, 1);
        // ms1 base is 2x ms0 base; allow for jitter of ±25% on each
        // ms0 base=10, ms1 base=20 — ms1 must be > ms0 base (10) even with zero jitter
        assert!(ms0 >= 10, "attempt 0 should be at least initial_backoff_ms");
        assert!(
            ms1 >= 20,
            "attempt 1 should be at least 2x initial_backoff_ms"
        );
    }
}
