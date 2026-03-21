//! Test doubles for the command registry trait.
//!
//! Gated behind `feature = "test-support"`. Available to downstream crates
//! via `weft_commands = { ..., features = ["test-support"] }` in `[dev-dependencies]`.
//!
//! **Available stubs:**
//! - [`StubCommandRegistry`] — configurable success/failure behavior
//! - [`reconstruct_command_error`] — helper to reconstruct a `CommandError` from a `&CommandError`

use async_trait::async_trait;
use weft_core::{CommandDescription, CommandInvocation, CommandResult, CommandStub};

use crate::{CommandError, CommandRegistry};

// ── StubCommandRegistry ───────────────────────────────────────────────────────

/// A stub command registry with configurable success/failure behavior.
///
/// By default, `list_commands` returns a single stub command and `execute_command`
/// returns a successful result. The registry can be configured to:
/// - Fail a specific command with a `CommandError` (infrastructure failure).
/// - Return a `CommandResult { success: false }` for a specific command.
/// - Fail `list_commands` entirely with `RegistryUnavailable`.
pub struct StubCommandRegistry {
    /// If set, `execute_command` for the matching name returns this error.
    ///
    /// `CommandError` does not implement `Clone`. The error is stored here and
    /// reconstructed via [`reconstruct_command_error`] on each call.
    pub failing_command: Option<(String, CommandError)>,
    /// If set, `execute_command` for the matching name returns `CommandResult { success: false }`.
    pub failed_result_command: Option<(String, String)>,
    /// If true, `list_commands` returns a `RegistryUnavailable` error.
    pub fail_list_commands: bool,
}

impl StubCommandRegistry {
    /// Construct a registry where all operations succeed.
    pub fn new() -> Self {
        Self {
            failing_command: None,
            failed_result_command: None,
            fail_list_commands: false,
        }
    }

    /// Construct a registry where `execute_command` for `command_name` returns `error`.
    pub fn with_failing_command(command_name: impl Into<String>, error: CommandError) -> Self {
        Self {
            failing_command: Some((command_name.into(), error)),
            failed_result_command: None,
            fail_list_commands: false,
        }
    }

    /// Construct a registry where `execute_command` for `command_name` returns `success: false`.
    pub fn with_failed_result(
        command_name: impl Into<String>,
        error_msg: impl Into<String>,
    ) -> Self {
        Self {
            failing_command: None,
            failed_result_command: Some((command_name.into(), error_msg.into())),
            fail_list_commands: false,
        }
    }

    /// Construct a registry where `list_commands` returns `RegistryUnavailable`.
    pub fn with_list_commands_failing() -> Self {
        Self {
            failing_command: None,
            failed_result_command: None,
            fail_list_commands: true,
        }
    }
}

impl Default for StubCommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CommandRegistry for StubCommandRegistry {
    async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
        if self.fail_list_commands {
            return Err(CommandError::RegistryUnavailable(
                "stub: registry unavailable".to_string(),
            ));
        }
        Ok(vec![CommandStub {
            name: "test_command".to_string(),
            description: "A test command".to_string(),
        }])
    }

    async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
        Ok(CommandDescription {
            name: name.to_string(),
            description: format!("{name}: test command"),
            usage: format!("/{name}"),
            parameters_schema: None,
        })
    }

    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, CommandError> {
        // Check if this command is configured to fail with an infrastructure error.
        if let Some((ref failing_name, ref err)) = self.failing_command
            && invocation.name == *failing_name
        {
            // CommandError doesn't implement Clone — reconstruct an equivalent error.
            return Err(reconstruct_command_error(err));
        }

        // Check if this command is configured to return a failed result (success=false).
        if let Some((ref failing_name, ref error_msg)) = self.failed_result_command
            && invocation.name == *failing_name
        {
            return Ok(CommandResult {
                command_name: invocation.name.clone(),
                success: false,
                output: String::new(),
                error: Some(error_msg.clone()),
            });
        }

        // Default: return a successful result.
        Ok(CommandResult {
            command_name: invocation.name.clone(),
            success: true,
            output: format!("stub output for {}", invocation.name),
            error: None,
        })
    }
}

// ── reconstruct_command_error ─────────────────────────────────────────────────

/// Reconstruct a `CommandError` from a stored reference.
///
/// `CommandError` does not implement `Clone`. This helper reconstructs an equivalent
/// error from the stored value for use in test stubs where the error must be returned
/// from a `&self` method (e.g., `execute_command` in `StubCommandRegistry`).
pub fn reconstruct_command_error(err: &CommandError) -> CommandError {
    match err {
        CommandError::NotFound(s) => CommandError::NotFound(s.clone()),
        CommandError::ExecutionFailed { name, reason } => CommandError::ExecutionFailed {
            name: name.clone(),
            reason: reason.clone(),
        },
        CommandError::InvalidArguments { name, reason } => CommandError::InvalidArguments {
            name: name.clone(),
            reason: reason.clone(),
        },
        CommandError::RegistryUnavailable(s) => CommandError::RegistryUnavailable(s.clone()),
    }
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use weft_core::CommandAction;

    fn make_invocation(name: &str) -> CommandInvocation {
        CommandInvocation {
            name: name.to_string(),
            action: CommandAction::Execute,
            arguments: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn stub_command_registry_lists_commands() {
        let registry = StubCommandRegistry::new();
        let cmds = registry.list_commands().await.unwrap();
        assert!(!cmds.is_empty());
        assert_eq!(cmds[0].name, "test_command");
    }

    #[tokio::test]
    async fn stub_command_registry_executes_successfully() {
        let registry = StubCommandRegistry::new();
        let result = registry
            .execute_command(&make_invocation("any"))
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.command_name, "any");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn stub_command_registry_returns_error_for_configured_command() {
        let registry = StubCommandRegistry::with_failing_command(
            "bad_cmd",
            CommandError::RegistryUnavailable("down".to_string()),
        );
        let result = registry.execute_command(&make_invocation("bad_cmd")).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CommandError::RegistryUnavailable(_)
        ));
    }

    #[tokio::test]
    async fn stub_command_registry_other_commands_succeed_when_one_fails() {
        let registry = StubCommandRegistry::with_failing_command(
            "bad_cmd",
            CommandError::NotFound("bad_cmd".to_string()),
        );
        // A different command should still succeed.
        let result = registry.execute_command(&make_invocation("good_cmd")).await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }

    #[tokio::test]
    async fn stub_command_registry_returns_failed_result_when_configured() {
        let registry = StubCommandRegistry::with_failed_result("partial_cmd", "quota exceeded");
        let result = registry
            .execute_command(&make_invocation("partial_cmd"))
            .await
            .unwrap();
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("quota exceeded"));
    }

    #[tokio::test]
    async fn stub_command_registry_list_commands_fails_when_configured() {
        let registry = StubCommandRegistry::with_list_commands_failing();
        let result = registry.list_commands().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CommandError::RegistryUnavailable(_)
        ));
    }

    #[tokio::test]
    async fn stub_command_registry_describes_any_command() {
        let registry = StubCommandRegistry::new();
        let desc = registry.describe_command("my_cmd").await.unwrap();
        assert_eq!(desc.name, "my_cmd");
        assert!(desc.usage.contains("my_cmd"));
    }

    #[test]
    fn reconstruct_command_error_not_found() {
        let original = CommandError::NotFound("cmd".to_string());
        let reconstructed = reconstruct_command_error(&original);
        assert!(matches!(reconstructed, CommandError::NotFound(s) if s == "cmd"));
    }

    #[test]
    fn reconstruct_command_error_execution_failed() {
        let original = CommandError::ExecutionFailed {
            name: "x".to_string(),
            reason: "oops".to_string(),
        };
        let reconstructed = reconstruct_command_error(&original);
        assert!(matches!(
            reconstructed,
            CommandError::ExecutionFailed { name, reason } if name == "x" && reason == "oops"
        ));
    }

    #[test]
    fn reconstruct_command_error_invalid_arguments() {
        let original = CommandError::InvalidArguments {
            name: "x".to_string(),
            reason: "bad arg".to_string(),
        };
        let reconstructed = reconstruct_command_error(&original);
        assert!(matches!(
            reconstructed,
            CommandError::InvalidArguments { name, reason } if name == "x" && reason == "bad arg"
        ));
    }

    #[test]
    fn reconstruct_command_error_registry_unavailable() {
        let original = CommandError::RegistryUnavailable("down".to_string());
        let reconstructed = reconstruct_command_error(&original);
        assert!(matches!(
            reconstructed,
            CommandError::RegistryUnavailable(s) if s == "down"
        ));
    }
}
