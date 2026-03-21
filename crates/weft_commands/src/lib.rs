//! `weft_commands` — Command registry trait, parser, and adapter.
//!
//! Contains:
//! - `CommandRegistry` trait for managing available commands
//! - `CommandError` error type
//! - `parser`: slash command parser for LLM output
//! - `adapter`: `ToolRegistryCommandAdapter` bridging `ToolRegistryClient` to `CommandRegistry`
//!
//! Tool client types have moved to `weft_tools`.
//! Memory store types and clients have moved to `weft_memory`.

pub mod adapter;
pub mod parser;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use adapter::ToolRegistryCommandAdapter;
pub use parser::{ParsedResponse, parse_response};

// Re-export tool types from weft_tools for consumers that import via weft_commands.
pub use weft_tools::{
    GrpcToolRegistryClient, ToolDescription, ToolExecutionResult, ToolInfo, ToolRegistryClient,
    ToolRegistryError,
};

// Re-export memory types from weft_memory for backward compatibility.
pub use weft_memory::{
    GrpcMemoryStoreClient, MemoryEntry, MemoryQueryResult, MemoryStoreClient, MemoryStoreError,
    MemoryStoreMux, MemoryStoreResult,
};

use async_trait::async_trait;
use weft_core::{CommandDescription, CommandInvocation, CommandResult, CommandStub};

/// Registry of available commands. In v1, populated from a single ToolRegistry.
#[async_trait]
pub trait CommandRegistry: Send + Sync + 'static {
    /// List all registered command stubs.
    async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError>;

    /// Get the full description of a command (progressive disclosure).
    async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError>;

    /// Execute a command with the given arguments.
    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, CommandError>;
}

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
