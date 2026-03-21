//! Test hooks: unit test access to private free functions.
//!
//! This module re-exports private functions from sibling modules for use
//! in unit tests. Only compiled when `test` or the `test-support` feature
//! is enabled.

use super::retry::{backoff_ms, should_retry};
use crate::budget::Budget;
use crate::config::RetryPolicy;
use crate::error::ReactorError;
use crate::event::Event;
use crate::execution::ExecutionId;
use tokio_util::sync::CancellationToken;

use super::Reactor;

/// Expose `should_retry` for unit testing.
pub fn should_retry_pub(
    policy: Option<&RetryPolicy>,
    attempt: u32,
    budget: &Budget,
    cancel: &CancellationToken,
) -> bool {
    should_retry(policy, attempt, budget, cancel)
}

/// Expose `backoff_ms` for unit testing.
pub fn backoff_ms_pub(policy: &RetryPolicy, attempt: u32) -> u64 {
    backoff_ms(policy, attempt)
}

/// Expose `Reactor::check_idempotency` for unit testing.
///
/// Allows tests to call the idempotency check directly with a known
/// execution_id rather than going through `execute()` which always creates
/// a fresh id.
pub async fn check_idempotency_pub(
    reactor: &Reactor,
    execution_id: &ExecutionId,
    key: &str,
) -> Result<Option<Vec<Event>>, ReactorError> {
    reactor.check_idempotency(execution_id, key).await
}
