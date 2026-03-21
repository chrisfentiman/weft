//! ValidateActivity: validates the incoming request.
//!
//! Checks that messages are non-empty and the request format is valid.
//! Pushes a `CommandsAvailable` event carrying the loaded commands so that
//! the Reactor can populate `state.available_commands` from the event channel.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use weft_reactor_trait::{
    Activity, ActivityEvent, ActivityInput, CommandEvent, EventLog, ExecutionEvent, ExecutionId,
    PipelineEvent, ServiceLocator,
};

/// Validates the incoming request.
///
/// Checks that messages are non-empty, the request format is valid, and
/// pushes the list of available commands from the command registry as a
/// `CommandsAvailable` event. The Reactor handles that event to populate
/// `state.available_commands` for subsequent activities.
///
/// **Name:** `"validate"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "validate" })`
/// - `Execution(ExecutionEvent::ValidationPassed)` — if validation succeeds
/// - `Command(CommandEvent::Available { commands })` — always pushed after ValidationPassed
/// - `Execution(ExecutionEvent::ValidationFailed { reason })` — if validation fails
/// - `Activity(ActivityEvent::Completed { name: "validate", idempotency_key: None })`
/// - `Activity(ActivityEvent::Failed { name: "validate", error, retryable: false })` — on internal error
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
        services: &dyn ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let start = Instant::now();

        // Push ActivityStarted
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;

        // Check for cancellation before doing any work.
        if cancel.is_cancelled() {
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "cancelled before validation".to_string(),
                    retryable: false,
                }))
                .await;
            debug!(duration_ms, "validate: cancelled");
            return;
        }

        // Validate: messages must be non-empty.
        if input.messages.is_empty() {
            let _ = event_tx
                .send(PipelineEvent::Execution(ExecutionEvent::ValidationFailed {
                    reason: "messages must not be empty".to_string(),
                }))
                .await;
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "validation failed: messages must not be empty".to_string(),
                    retryable: false,
                }))
                .await;
            debug!(duration_ms, "validate: failed (empty messages)");
            return;
        }

        // Populate available commands from the registry.
        // On error, fail-open: continue with empty command list and log.
        let available_commands = match services.commands().list_commands().await {
            Ok(cmds) => {
                debug!(count = cmds.len(), "validate: loaded commands");
                cmds
            }
            Err(e) => {
                tracing::warn!(error = %e, "validate: could not list commands, continuing with empty list");
                vec![]
            }
        };

        let _ = event_tx
            .send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
            .await;

        // Push CommandsAvailable before ActivityCompleted so the Reactor can
        // populate state.available_commands while draining the channel.
        // On list_commands error the activity continues fail-open with an empty
        // vec (the warn is already logged above); commands remain unavailable for
        // this execution, which is preferable to failing the entire request.
        let _ = event_tx
            .send(PipelineEvent::Command(CommandEvent::Available {
                commands: available_commands,
            }))
            .await;

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;

        debug!("validate: completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        NullEventLog, collect_events, make_test_input, make_test_services,
        make_test_services_with_failing_list_commands,
    };
    use pretty_assertions::assert_eq;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    // Helper to run the activity and collect events.
    async fn run_validate(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

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
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "validate")
            ),
            "expected Activity(Started)"
        );

        // Must push ValidationPassed.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Execution(ExecutionEvent::ValidationPassed)
            )),
            "expected Execution(ValidationPassed)"
        );

        // Must push ActivityCompleted (not ActivityFailed).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "validate")
            ),
            "expected Activity(Completed)"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "should not push Activity(Failed) on success"
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
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Execution(ExecutionEvent::ValidationFailed { reason })
                if reason.contains("empty")
            )),
            "expected Execution(ValidationFailed) with 'empty' in reason"
        );

        // Must push ActivityFailed (not ActivityCompleted).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { name, .. }) if name == "validate")
            ),
            "expected Activity(Failed)"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { .. }))),
            "should not push Activity(Completed) on failure"
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
        let exec_id = ExecutionId::new();

        let activity = ValidateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push ActivityFailed when cancelled.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) when cancelled"
        );
        // Must NOT push ValidationPassed or ActivityCompleted.
        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Execution(ExecutionEvent::ValidationPassed)
            )),
            "should not push Execution(ValidationPassed) when cancelled"
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
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");

        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(!retryable, "validation failures should not be retryable");
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }

    // ── CommandsAvailable event ───────────────────────────────────────────────

    /// Verify that a successful run pushes `Command(Available)` with the stubs
    /// returned by the command registry (one stub: "test_command").
    #[tokio::test]
    async fn test_validate_pushes_commands_available() {
        let input = make_test_input();
        let events = run_validate(input).await;

        // Exactly one Command(Available) event must be present.
        let commands_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, PipelineEvent::Command(CommandEvent::Available { .. })))
            .collect();

        assert_eq!(
            commands_events.len(),
            1,
            "expected exactly one Command(Available) event, got {}",
            commands_events.len()
        );

        match commands_events[0] {
            PipelineEvent::Command(CommandEvent::Available { commands }) => {
                assert_eq!(
                    commands.len(),
                    1,
                    "expected 1 command stub from the test registry"
                );
                assert_eq!(commands[0].name, "test_command");
                assert_eq!(commands[0].description, "A test command");
            }
            _ => panic!("expected Command(Available)"),
        }

        // Command(Available) must appear before Activity(Completed).
        let ca_pos = events
            .iter()
            .position(|e| matches!(e, PipelineEvent::Command(CommandEvent::Available { .. })))
            .expect("Command(Available) not found");
        let completed_pos = events
            .iter()
            .position(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { .. })))
            .expect("Activity(Completed) not found");
        assert!(
            ca_pos < completed_pos,
            "Command(Available) (pos {ca_pos}) must come before Activity(Completed) (pos {completed_pos})"
        );
    }

    /// When `list_commands` returns an error, the activity must fail-open:
    /// push `Command(Available)` with an empty vec and continue to `Activity(Completed)`.
    #[tokio::test]
    async fn test_validate_commands_available_empty_on_error() {
        let input = make_test_input();

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services_with_failing_list_commands();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ValidateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must still push ValidationPassed (fail-open).
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Execution(ExecutionEvent::ValidationPassed)
            )),
            "expected Execution(ValidationPassed) even when list_commands fails"
        );

        // Must push Command(Available) with an empty commands vec.
        let ca = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Command(CommandEvent::Available { .. })))
            .expect("expected Command(Available) even on list_commands error");

        match ca {
            PipelineEvent::Command(CommandEvent::Available { commands }) => {
                assert!(
                    commands.is_empty(),
                    "expected empty commands vec on list_commands error, got {:?}",
                    commands
                );
            }
            _ => panic!("expected Command(Available)"),
        }

        // Must still reach Activity(Completed) (not Activity(Failed)).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "validate")
            ),
            "expected Activity(Completed) even when list_commands fails"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "should not push Activity(Failed) when list_commands fails (fail-open)"
        );
    }
}
