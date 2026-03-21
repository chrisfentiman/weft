//! `weft_commands_trait` — Command registry trait and error type.
//!
//! Contains the `CommandRegistry` trait. The implementations live in
//! `weft_commands`, which depends on this crate.
//!
//! Consumers that need the trait boundary without the implementation (e.g.
//! `weft_reactor_trait`) depend on this crate directly.

use async_trait::async_trait;
use weft_core::{CommandDescription, CommandInvocation, CommandResult, CommandStub};

// ── CommandError ───────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CommandError {
    #[error("command not found: {0}")]
    NotFound(String),
    #[error("command execution failed: {name}: {reason}")]
    ExecutionFailed { name: String, reason: String },
    #[error("invalid arguments for {name}: {reason}")]
    InvalidArguments { name: String, reason: String },
    #[error("registry unavailable: {0}")]
    RegistryUnavailable(String),
}

// ── CommandRegistry trait ──────────────────────────────────────────────────

/// Registry of available commands.
///
/// `Send + Sync + 'static`: shared via Arc across async request handlers.
#[async_trait]
pub trait CommandRegistry: Send + Sync + 'static {
    async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError>;
    async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError>;
    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, CommandError>;
}
