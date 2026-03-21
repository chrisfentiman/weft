//! ExecuteCommandActivity: executes a single command invocation.
//!
//! Calls `services.commands().execute_command()` with the `CommandInvocation` from
//! `ActivityInput.metadata`. Pushes CommandStarted and CommandCompleted events.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use weft_commands::CommandError;
use weft_core::CommandInvocation;

use weft_reactor_trait::{
    Activity, ActivityEvent, ActivityInput, CommandEvent, EventLog, ExecutionId, PipelineEvent,
    ServiceLocator,
};

/// Executes a single command invocation via the command registry.
///
/// The command invocation is read from `input.metadata["invocation"]` as a
/// serialized [`weft_core::CommandInvocation`]. The Reactor populates this field
/// before invoking the activity.
///
/// **Note:** Command failures are NOT activity failures. A failed command pushes
/// `Command(CommandEvent::Failed)` (with error details) and then
/// `Command(CommandEvent::Completed)` (with `success: false`). Only infrastructure
/// failures push `Activity(ActivityEvent::Failed)`.
///
/// **Name:** `"execute_command"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "execute_command" })`
/// - `Command(CommandEvent::Started { invocation })` — the full CommandInvocation
/// - `Command(CommandEvent::Completed { name, result })` — with full CommandResult (success or failure)
/// - `Command(CommandEvent::Failed { name, error })` — if the command failed (in addition to Completed)
/// - `Activity(ActivityEvent::Completed { name: "execute_command", idempotency_key })` — always
/// - `Activity(ActivityEvent::Failed { name: "execute_command", error, retryable })` — only on infrastructure error
pub struct ExecuteCommandActivity;

impl ExecuteCommandActivity {
    /// Construct a new ExecuteCommandActivity.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExecuteCommandActivity {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a command error as retryable.
///
/// Infrastructure errors (registry unavailable) are retryable.
/// Logic errors (not found, invalid arguments, execution failed) are not.
fn command_error_retryable(err: &CommandError) -> bool {
    matches!(err, CommandError::RegistryUnavailable(_))
}

#[async_trait::async_trait]
impl Activity for ExecuteCommandActivity {
    fn name(&self) -> &str {
        "execute_command"
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
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;

        if cancel.is_cancelled() {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "cancelled before command execution".to_string(),
                    retryable: false,
                }))
                .await;
            return;
        }

        // Extract the CommandInvocation from metadata.
        let invocation = match extract_invocation(&input) {
            Ok(inv) => inv,
            Err(e) => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: format!("invalid invocation in metadata: {e}"),
                        retryable: false,
                    }))
                    .await;
                return;
            }
        };

        let command_name = invocation.name.clone();
        debug!(command = %command_name, "execute_command: starting");

        // Push Command(Started) with full invocation.
        let _ = event_tx
            .send(PipelineEvent::Command(CommandEvent::Started {
                invocation: invocation.clone(),
            }))
            .await;

        // Execute the command.
        let command_result = tokio::select! {
            result = services.commands().execute_command(&invocation) => result,
            _ = cancel.cancelled() => {
                let _ = event_tx
                    .send(PipelineEvent::Command(CommandEvent::Failed {
                        name: command_name.clone(),
                        error: "cancelled during command execution".to_string(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: "cancelled during command execution".to_string(),
                        retryable: false,
                    }))
                    .await;
                return;
            }
        };

        match command_result {
            Ok(result) => {
                debug!(command = %command_name, success = result.success, "execute_command: completed");
                // If command reported failure, also push Command(Failed) for observability.
                if !result.success {
                    let error_msg = result
                        .error
                        .clone()
                        .unwrap_or_else(|| "command reported failure".to_string());
                    let _ = event_tx
                        .send(PipelineEvent::Command(CommandEvent::Failed {
                            name: command_name.clone(),
                            error: error_msg,
                        }))
                        .await;
                }
                // Always push Command(Completed) with the full result.
                let _ = event_tx
                    .send(PipelineEvent::Command(CommandEvent::Completed {
                        name: command_name.clone(),
                        result,
                    }))
                    .await;
            }
            Err(e) => {
                // Infrastructure failure (registry unavailable, etc.).
                let retryable = command_error_retryable(&e);
                debug!(command = %command_name, error = %e, retryable, "execute_command: infrastructure error");
                let _ = event_tx
                    .send(PipelineEvent::Command(CommandEvent::Failed {
                        name: command_name.clone(),
                        error: e.to_string(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: e.to_string(),
                        retryable,
                    }))
                    .await;
                // No Activity(Completed) on infrastructure failure.
                return;
            }
        }

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;

        debug!(command = %command_name, "execute_command: activity completed");
    }
}

/// Extract a CommandInvocation from ActivityInput metadata.
///
/// The Reactor stores the invocation as `input.metadata["invocation"]` before
/// calling this activity. Returns an error if the field is missing or malformed.
fn extract_invocation(input: &ActivityInput) -> Result<CommandInvocation, String> {
    let inv_value = input
        .metadata
        .get("invocation")
        .ok_or("missing 'invocation' field in metadata")?;
    serde_json::from_value(inv_value.clone())
        .map_err(|e| format!("failed to deserialize invocation: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        NullEventLog, collect_events, make_test_input, make_test_services,
        make_test_services_with_failed_command,
    };
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_core::{CommandAction, CommandInvocation};

    fn make_input_with_invocation(name: &str) -> ActivityInput {
        let mut input = make_test_input();
        let invocation = CommandInvocation {
            name: name.to_string(),
            action: CommandAction::Execute,
            arguments: serde_json::json!({}),
        };
        input.metadata = serde_json::json!({
            "invocation": serde_json::to_value(&invocation).unwrap()
        });
        input.idempotency_key = Some(format!("exec:execute_command:0:{name}"));
        input
    }

    async fn run_execute(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ExecuteCommandActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn execute_command_name() {
        assert_eq!(ExecuteCommandActivity::new().name(), "execute_command");
    }

    // ── Happy path ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn execute_command_pushes_command_started_and_completed() {
        let input = make_input_with_invocation("test_command");
        let events = run_execute(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "execute_command")),
            "expected Activity(Started)"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Command(CommandEvent::Started { invocation }) if invocation.name == "test_command")),
            "expected Command(Started) with correct invocation"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Command(CommandEvent::Completed { name, .. }) if name == "test_command")),
            "expected Command(Completed)"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "execute_command")),
            "expected Activity(Completed)"
        );
    }

    // ── Idempotency key in Activity(Completed) ─────────────────────────────

    #[tokio::test]
    async fn execute_command_includes_idempotency_key_in_completed() {
        let input = make_input_with_invocation("test_command");
        let expected_key = input.idempotency_key.clone().unwrap();
        let events = run_execute(input).await;

        let completed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "execute_command"))
            .expect("expected Activity(Completed)");

        match completed {
            PipelineEvent::Activity(ActivityEvent::Completed {
                idempotency_key, ..
            }) => {
                assert_eq!(*idempotency_key, Some(expected_key));
            }
            _ => panic!("expected Activity(Completed)"),
        }
    }

    // ── Missing invocation in metadata ────────────────────────────────────

    #[tokio::test]
    async fn execute_command_fails_with_missing_invocation() {
        let input = make_test_input(); // No invocation in metadata
        let events = run_execute(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) when invocation is missing"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { .. }))),
            "should not push Activity(Completed) on missing invocation"
        );
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn execute_command_handles_cancellation() {
        let input = make_input_with_invocation("test_command");
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ExecuteCommandActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) on cancellation"
        );
    }

    // ── Infrastructure failure retryable ─────────────────────────────────

    #[tokio::test]
    async fn execute_command_infrastructure_failure_is_retryable() {
        let services = crate::test_support::make_test_services_with_command_error(
            "test_command",
            CommandError::RegistryUnavailable("registry down".to_string()),
        );
        let input = make_input_with_invocation("test_command");
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ExecuteCommandActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");
        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(*retryable, "registry unavailable should be retryable");
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }

    // ── command_error_retryable ──────────────────────────────────────────

    #[test]
    fn registry_unavailable_is_retryable() {
        let err = CommandError::RegistryUnavailable("down".to_string());
        assert!(command_error_retryable(&err));
    }

    #[test]
    fn command_not_found_not_retryable() {
        let err = CommandError::NotFound("cmd".to_string());
        assert!(!command_error_retryable(&err));
    }

    #[test]
    fn command_execution_failed_not_retryable() {
        let err = CommandError::ExecutionFailed {
            name: "cmd".to_string(),
            reason: "reason".to_string(),
        };
        assert!(!command_error_retryable(&err));
    }

    // ── Command failure result (success=false) ───────────────────────────

    #[tokio::test]
    async fn execute_command_failed_result_pushes_command_completed_with_failure() {
        // When the command registry returns CommandResult { success: false },
        // the activity must push Command(Failed) (for observability) and then
        // Command(Completed) (with the failure result). Activity(Completed) must
        // still be pushed — a failed command result is not an infrastructure error.
        let services =
            make_test_services_with_failed_command("failing_cmd", "command error detail");
        let input = make_input_with_invocation("failing_cmd");
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ExecuteCommandActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Command(Failed) must be pushed (observability event for failed results).
        let cmd_failed = events.iter().find(
            |e| matches!(e, PipelineEvent::Command(CommandEvent::Failed { name, .. }) if name == "failing_cmd"),
        );
        assert!(
            cmd_failed.is_some(),
            "expected Command(Failed) when command returns success=false"
        );

        // Command(Completed) must be pushed with success=false.
        let cmd_completed = events
            .iter()
            .find(|e| {
                matches!(e, PipelineEvent::Command(CommandEvent::Completed { name, .. }) if name == "failing_cmd")
            })
            .expect("expected Command(Completed) after failed command result");
        match cmd_completed {
            PipelineEvent::Command(CommandEvent::Completed { result, .. }) => {
                assert!(
                    !result.success,
                    "Command(Completed) result must have success=false"
                );
                assert_eq!(
                    result.error.as_deref(),
                    Some("command error detail"),
                    "Command(Completed) result must carry the error message"
                );
            }
            _ => panic!("expected Command(Completed)"),
        }

        // Activity(Completed) must still be pushed — failed command is not an
        // infrastructure error and does not abort the activity.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "execute_command")),
            "expected Activity(Completed) even when command returns success=false"
        );

        // Activity(Failed) must NOT be pushed.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "Activity(Failed) must not be pushed for a command-level failure result"
        );
    }
}
