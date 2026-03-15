//! `weft_commands` — Command registry trait, parser, gRPC client, and adapter.
//!
//! Contains:
//! - `CommandRegistry` trait for managing available commands
//! - `ToolRegistryClient` trait for communicating with a remote gRPC ToolRegistry
//! - `CommandError` and `ToolRegistryError` error types
//! - `parser`: slash command parser for LLM output
//! - `grpc_client`: tonic client implementing `ToolRegistryClient`
//! - `adapter`: `ToolRegistryCommandAdapter` bridging `ToolRegistryClient` to `CommandRegistry`
//! - `types`: gRPC mapping types (`ToolInfo`, `ToolDescription`, `ToolExecutionResult`)

pub mod adapter;
pub mod grpc_client;
pub mod parser;
pub mod types;

pub use adapter::ToolRegistryCommandAdapter;
pub use grpc_client::GrpcToolRegistryClient;
pub use parser::{ParsedResponse, parse_response};
pub use types::{ToolDescription, ToolExecutionResult, ToolInfo};

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

/// Client interface to a remote gRPC ToolRegistry service.
/// Co-located with commands because the only consumer is the ToolRegistryCommandAdapter.
#[async_trait]
pub trait ToolRegistryClient: Send + Sync + 'static {
    async fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolRegistryError>;

    async fn describe_tool(&self, name: &str) -> Result<ToolDescription, ToolRegistryError>;

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecutionResult, ToolRegistryError>;
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

#[derive(Debug, thiserror::Error)]
pub enum ToolRegistryError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("tool not found: {0}")]
    ToolNotFound(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("grpc error: {0}")]
    GrpcError(String),
}
