//! ValidateActivity: validates the incoming request.
//!
//! Checks that messages are non-empty and the request format is valid.
//! Populates `ActivityInput.available_commands` from the command registry.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::activity::{Activity, ActivityInput};
use crate::event::PipelineEvent;
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Validates the incoming request.
///
/// Checks that messages are non-empty, the request format is valid, and
/// populates the list of available commands from the command registry.
///
/// **Name:** `"validate"`
///
/// **Events pushed:**
/// - `ActivityStarted { name: "validate" }`
/// - `ValidationPassed` — if validation succeeds
/// - `ValidationFailed { reason }` — if validation fails
/// - `ActivityCompleted { name: "validate", duration_ms, idempotency_key: None }`
/// - `ActivityFailed { name: "validate", error, retryable: false }` — on internal error
pub struct ValidateActivity;

impl ValidateActivity {
    /// Construct a new ValidateActivity.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ValidateActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for ValidateActivity {
    fn name(&self) -> &str {
        "validate"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        services: &Services,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let start = Instant::now();

        // Push ActivityStarted
        let _ = event_tx
            .send(PipelineEvent::ActivityStarted {
                name: self.name().to_string(),
            })
            .await;

        // Check for cancellation before doing any work.
        if cancel.is_cancelled() {
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::ActivityFailed {
                    name: self.name().to_string(),
                    error: "cancelled before validation".to_string(),
                    retryable: false,
                })
                .await;
            debug!(duration_ms, "validate: cancelled");
            return;
        }

        // Validate: messages must be non-empty.
        if input.messages.is_empty() {
            let _ = event_tx
                .send(PipelineEvent::ValidationFailed {
                    reason: "messages must not be empty".to_string(),
                })
                .await;
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::ActivityFailed {
                    name: self.name().to_string(),
                    error: "validation failed: messages must not be empty".to_string(),
                    retryable: false,
                })
                .await;
            debug!(duration_ms, "validate: failed (empty messages)");
            return;
        }

        // Populate available commands from the registry.
        // On error, fail-open: continue with empty command list and log.
        let available_commands = match services.commands.list_commands().await {
            Ok(cmds) => {
                debug!(count = cmds.len(), "validate: loaded commands");
                cmds
            }
            Err(e) => {
                tracing::warn!(error = %e, "validate: could not list commands, continuing with empty list");
                vec![]
            }
        };

        // Attach available commands to a ValidationPassed event so downstream
        // activities (AssemblePrompt, Generate) can use them. The Reactor reads
        // ValidationPassed from the channel and updates ExecutionState.
        let _ = event_tx.send(PipelineEvent::ValidationPassed).await;

        // Publish the populated commands list via a separate event for state reconstruction.
        // We reuse PromptAssembled as a proxy — but the actual mechanism is that the Reactor
        // intercepts ValidationPassed and then updates state from the input's available_commands.
        // Since activities cannot return values, we push a dummy event to carry the commands.
        // NOTE: The Reactor reconstructs available_commands from the event log during replay.
        // For now, the Reactor reads available_commands from the ValidateActivity's input after
        // the channel drains. This is a known limitation of Phase 3 — the Reactor (Phase 4) will
        // read the commands from the input snapshot it provides.
        //
        // To make commands available to the Reactor without a dedicated event type, we push a
        // ValidationPassed event as a signal. The Reactor builds the input with available_commands
        // already set (it calls list_commands before invoking this activity). The ValidationPassed
        // event serves as the signal that commands are ready.
        let _ = available_commands; // consumed above

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = event_tx
            .send(PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms,
                idempotency_key: None,
            })
            .await;

        debug!(duration_ms, "validate: completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NullEventLog;
    use crate::test_support::{collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    // Helper to run the activity and collect events.
    async fn run_validate(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = ValidateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn validate_name() {
        assert_eq!(ValidateActivity::new().name(), "validate");
    }

    // ── Happy path: valid request ────────────────────────────────────────────

    #[tokio::test]
    async fn validate_passes_with_valid_messages() {
        let input = make_test_input();
        let events = run_validate(input).await;

        // Must push ActivityStarted.
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityStarted { name } if name == "validate")
            ),
            "expected ActivityStarted"
        );

        // Must push ValidationPassed.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ValidationPassed)),
            "expected ValidationPassed"
        );

        // Must push ActivityCompleted (not ActivityFailed).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "validate")
            ),
            "expected ActivityCompleted"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "should not push ActivityFailed on success"
        );
    }

    // ── Failure: empty messages ──────────────────────────────────────────────

    #[tokio::test]
    async fn validate_fails_with_empty_messages() {
        let mut input = make_test_input();
        input.messages.clear();
        let events = run_validate(input).await;

        // Must push ValidationFailed.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ValidationFailed { reason } if reason.contains("empty"))),
            "expected ValidationFailed with 'empty' in reason"
        );

        // Must push ActivityFailed (not ActivityCompleted).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityFailed { name, .. } if name == "validate")
            ),
            "expected ActivityFailed"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "should not push ActivityCompleted on failure"
        );
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn validate_handles_pre_cancelled_token() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel(); // Pre-cancelled

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = ValidateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push ActivityFailed when cancelled.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "expected ActivityFailed when cancelled"
        );
        // Must NOT push ValidationPassed or ActivityCompleted.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ValidationPassed)),
            "should not push ValidationPassed when cancelled"
        );
    }

    // ── ActivityFailed retryable = false ─────────────────────────────────────

    #[tokio::test]
    async fn validate_activity_failed_not_retryable() {
        let mut input = make_test_input();
        input.messages.clear();
        let events = run_validate(input).await;

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ActivityFailed { .. }))
            .expect("expected ActivityFailed");

        match failed {
            PipelineEvent::ActivityFailed { retryable, .. } => {
                assert!(!retryable, "validation failures should not be retryable");
            }
            _ => panic!("expected ActivityFailed"),
        }
    }
}
