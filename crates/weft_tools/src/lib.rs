//! `weft_tools` — ToolRegistry gRPC client, types, and client trait.
//!
//! Contains:
//! - `ToolRegistryClient` trait for communicating with a remote gRPC ToolRegistry
//! - `ToolRegistryError` error type
//! - `client`: `GrpcToolRegistryClient` — tonic implementation of `ToolRegistryClient`
//! - `types`: domain types (`ToolInfo`, `ToolDescription`, `ToolExecutionResult`)
//!
//! `weft_commands` depends on this crate; the adapter in `weft_commands` bridges
//! `ToolRegistryClient` to `CommandRegistry`.

pub mod client;
pub mod types;

pub use client::GrpcToolRegistryClient;
pub use types::{ToolDescription, ToolExecutionResult, ToolInfo};

use async_trait::async_trait;

/// Client interface to a remote gRPC ToolRegistry service.
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
