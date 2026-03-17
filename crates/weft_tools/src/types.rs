//! Domain types for the ToolRegistry service.
//!
//! These types bridge the protobuf wire format and the rest of the tool system.
//! Consumed by `ToolRegistryClient` implementations and `ToolRegistryCommandAdapter`.

/// Summary of a tool from the remote registry.
/// Maps to the gRPC `ToolInfo` message.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Unique tool name within the registry.
    pub name: String,
    /// One-line description.
    pub description: String,
}

/// Full tool description from the remote registry.
/// Maps to the gRPC `DescribeToolResponse` message.
#[derive(Debug, Clone)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    /// Detailed usage instructions.
    pub usage: String,
    /// JSON schema for the tool's parameters. None if no schema.
    pub parameters_schema: Option<serde_json::Value>,
}

/// Result of executing a tool via the remote registry.
/// Maps to the gRPC `ExecuteToolResponse` message.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    pub success: bool,
    /// Result content (text).
    pub output: String,
    /// Error message if success is false.
    pub error: Option<String>,
}
