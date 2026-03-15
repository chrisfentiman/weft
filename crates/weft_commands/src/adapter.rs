//! `ToolRegistryCommandAdapter`: bridges `ToolRegistryClient` to `CommandRegistry`.
//!
//! Maps:
//! - `list_commands()` -> `list_tools()` -> `ToolInfo` -> `CommandStub`
//! - `describe_command(name)` -> `describe_tool(name)` -> `ToolDescription` -> `CommandDescription`
//! - `execute_command(invocation)`:
//!   - `CommandAction::Describe` -> `describe_tool(name)` -> formatted `CommandResult`
//!   - `CommandAction::Execute` -> `execute_tool(name, args)` -> `ToolExecutionResult` -> `CommandResult`

use std::sync::Arc;

use async_trait::async_trait;
use weft_core::{CommandAction, CommandDescription, CommandInvocation, CommandResult, CommandStub};

use crate::{CommandError, CommandRegistry, ToolRegistryClient, ToolRegistryError};

/// Implements `CommandRegistry` by delegating to a `ToolRegistryClient`.
///
/// Uses `Arc<dyn ToolRegistryClient>` so the adapter is object-safe and can be wired
/// with any runtime-selected client (gRPC, mock, etc.) without monomorphization.
pub struct ToolRegistryCommandAdapter {
    client: Arc<dyn ToolRegistryClient>,
}

impl ToolRegistryCommandAdapter {
    pub fn new(client: Arc<dyn ToolRegistryClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl CommandRegistry for ToolRegistryCommandAdapter {
    async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
        let tools = self
            .client
            .list_tools()
            .await
            .map_err(tool_registry_to_command_error)?;

        Ok(tools
            .into_iter()
            .map(|t| CommandStub {
                name: t.name,
                description: t.description,
            })
            .collect())
    }

    async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
        let tool = self.client.describe_tool(name).await.map_err(|e| match e {
            ToolRegistryError::ToolNotFound(n) => CommandError::NotFound(n),
            other => tool_registry_to_command_error(other),
        })?;

        Ok(CommandDescription {
            name: tool.name,
            description: tool.description,
            usage: tool.usage,
            parameters_schema: tool.parameters_schema,
        })
    }

    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, CommandError> {
        match invocation.action {
            CommandAction::Describe => {
                // Progressive disclosure: describe_tool and format as CommandResult
                let description =
                    self.client
                        .describe_tool(&invocation.name)
                        .await
                        .map_err(|e| match e {
                            ToolRegistryError::ToolNotFound(n) => CommandError::NotFound(n),
                            other => tool_registry_to_command_error(other),
                        })?;

                let output = format_description_output(
                    &description.name,
                    &description.usage,
                    &description.description,
                );

                Ok(CommandResult {
                    command_name: invocation.name.clone(),
                    success: true,
                    output,
                    error: None,
                })
            }
            CommandAction::Execute => {
                let result = self
                    .client
                    .execute_tool(&invocation.name, invocation.arguments.clone())
                    .await
                    .map_err(|e| match e {
                        ToolRegistryError::ToolNotFound(n) => CommandError::NotFound(n),
                        ToolRegistryError::ExecutionFailed(reason) => {
                            CommandError::ExecutionFailed {
                                name: invocation.name.clone(),
                                reason,
                            }
                        }
                        other => tool_registry_to_command_error(other),
                    })?;

                Ok(CommandResult {
                    command_name: invocation.name.clone(),
                    success: result.success,
                    output: result.output,
                    error: result.error,
                })
            }
        }
    }
}

/// Format a tool description into a human-readable output string for injection into context.
///
/// Produces a string like:
/// ```text
/// web_search: Search the web for current information
///
/// Usage: /web_search query: "search terms", limit: 10
///
/// Parameters:
/// ...
/// ```
fn format_description_output(name: &str, usage: &str, description: &str) -> String {
    if usage.is_empty() {
        format!("{name}: {description}")
    } else {
        format!("{name}: {description}\n\nUsage: {usage}")
    }
}

/// Convert a `ToolRegistryError` to a `CommandError`.
fn tool_registry_to_command_error(e: ToolRegistryError) -> CommandError {
    match e {
        ToolRegistryError::ConnectionFailed(msg) => CommandError::RegistryUnavailable(msg),
        ToolRegistryError::ToolNotFound(name) => CommandError::NotFound(name),
        ToolRegistryError::ExecutionFailed(msg) => CommandError::ExecutionFailed {
            name: String::new(),
            reason: msg,
        },
        ToolRegistryError::GrpcError(msg) => CommandError::RegistryUnavailable(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolDescription, ToolExecutionResult, ToolInfo, ToolRegistryError};
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::Arc;
    use weft_core::CommandAction;

    /// Mock client for testing the adapter.
    struct MockToolRegistryClient {
        tools: Vec<ToolInfo>,
        description: Option<ToolDescription>,
        execution_result: Option<ToolExecutionResult>,
        fail_with: Option<ToolRegistryError>,
    }

    impl MockToolRegistryClient {
        fn new_with_tools(tools: Vec<ToolInfo>) -> Self {
            Self {
                tools,
                description: None,
                execution_result: None,
                fail_with: None,
            }
        }

        fn with_description(mut self, desc: ToolDescription) -> Self {
            self.description = Some(desc);
            self
        }

        fn with_execution_result(mut self, result: ToolExecutionResult) -> Self {
            self.execution_result = Some(result);
            self
        }

        fn with_failure(mut self, err: ToolRegistryError) -> Self {
            self.fail_with = Some(err);
            self
        }
    }

    #[async_trait]
    impl ToolRegistryClient for MockToolRegistryClient {
        async fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolRegistryError> {
            if let Some(ref e) = self.fail_with {
                return Err(match e {
                    ToolRegistryError::ConnectionFailed(msg) => {
                        ToolRegistryError::ConnectionFailed(msg.clone())
                    }
                    _ => ToolRegistryError::GrpcError("mock error".to_string()),
                });
            }
            Ok(self.tools.clone())
        }

        async fn describe_tool(&self, _name: &str) -> Result<ToolDescription, ToolRegistryError> {
            if let Some(ref e) = self.fail_with {
                return Err(match e {
                    ToolRegistryError::ToolNotFound(n) => {
                        ToolRegistryError::ToolNotFound(n.clone())
                    }
                    _ => ToolRegistryError::GrpcError("mock error".to_string()),
                });
            }
            self.description
                .clone()
                .ok_or_else(|| ToolRegistryError::ToolNotFound("no mock description".to_string()))
        }

        async fn execute_tool(
            &self,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<ToolExecutionResult, ToolRegistryError> {
            if let Some(ref e) = self.fail_with {
                return Err(match e {
                    ToolRegistryError::ExecutionFailed(msg) => {
                        ToolRegistryError::ExecutionFailed(msg.clone())
                    }
                    _ => ToolRegistryError::GrpcError("mock error".to_string()),
                });
            }
            self.execution_result
                .clone()
                .ok_or_else(|| ToolRegistryError::ExecutionFailed("no mock result".to_string()))
        }
    }

    fn make_tool_info(name: &str, desc: &str) -> ToolInfo {
        ToolInfo {
            name: name.to_string(),
            description: desc.to_string(),
        }
    }

    #[tokio::test]
    async fn test_list_commands_maps_tool_info_to_command_stub() {
        let client = MockToolRegistryClient::new_with_tools(vec![
            make_tool_info("web_search", "Search the web"),
            make_tool_info("code_review", "Review code"),
        ]);
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let commands = adapter.list_commands().await.expect("should succeed");
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].name, "web_search");
        assert_eq!(commands[0].description, "Search the web");
        assert_eq!(commands[1].name, "code_review");
        assert_eq!(commands[1].description, "Review code");
    }

    #[tokio::test]
    async fn test_list_commands_registry_unavailable_on_connection_failure() {
        let client = MockToolRegistryClient::new_with_tools(vec![])
            .with_failure(ToolRegistryError::ConnectionFailed("refused".to_string()));
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let result = adapter.list_commands().await;
        assert!(matches!(result, Err(CommandError::RegistryUnavailable(_))));
    }

    #[tokio::test]
    async fn test_describe_command_maps_tool_description_to_command_description() {
        let tool_desc = ToolDescription {
            name: "web_search".to_string(),
            description: "Search the web for current information".to_string(),
            usage: "/web_search query: \"search terms\", limit: 10".to_string(),
            parameters_schema: Some(json!({
                "type": "object",
                "properties": {"query": {"type": "string"}}
            })),
        };
        let client =
            MockToolRegistryClient::new_with_tools(vec![]).with_description(tool_desc.clone());
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let cmd_desc = adapter
            .describe_command("web_search")
            .await
            .expect("should succeed");
        assert_eq!(cmd_desc.name, "web_search");
        assert_eq!(
            cmd_desc.description,
            "Search the web for current information"
        );
        assert_eq!(
            cmd_desc.usage,
            "/web_search query: \"search terms\", limit: 10"
        );
        assert!(cmd_desc.parameters_schema.is_some());
    }

    #[tokio::test]
    async fn test_describe_command_not_found() {
        let client = MockToolRegistryClient::new_with_tools(vec![])
            .with_failure(ToolRegistryError::ToolNotFound("no_such".to_string()));
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let result = adapter.describe_command("no_such").await;
        assert!(matches!(result, Err(CommandError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_execute_command_execute_action() {
        let exec_result = ToolExecutionResult {
            success: true,
            output: "Found 3 results".to_string(),
            error: None,
        };
        let client =
            MockToolRegistryClient::new_with_tools(vec![]).with_execution_result(exec_result);
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let invocation = CommandInvocation {
            name: "web_search".to_string(),
            action: CommandAction::Execute,
            arguments: json!({"query": "Rust async"}),
        };
        let result = adapter
            .execute_command(&invocation)
            .await
            .expect("should succeed");
        assert!(result.success);
        assert_eq!(result.command_name, "web_search");
        assert_eq!(result.output, "Found 3 results");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_execute_command_describe_action() {
        let tool_desc = ToolDescription {
            name: "web_search".to_string(),
            description: "Search the web for current information".to_string(),
            usage: "/web_search query: \"search terms\"".to_string(),
            parameters_schema: None,
        };
        let client = MockToolRegistryClient::new_with_tools(vec![]).with_description(tool_desc);
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let invocation = CommandInvocation {
            name: "web_search".to_string(),
            action: CommandAction::Describe,
            arguments: serde_json::Value::Null,
        };
        let result = adapter
            .execute_command(&invocation)
            .await
            .expect("should succeed");
        assert!(result.success);
        assert_eq!(result.command_name, "web_search");
        assert!(result.output.contains("web_search"));
    }

    #[tokio::test]
    async fn test_execute_command_failed_execution() {
        let exec_result = ToolExecutionResult {
            success: false,
            output: String::new(),
            error: Some("network error".to_string()),
        };
        let client =
            MockToolRegistryClient::new_with_tools(vec![]).with_execution_result(exec_result);
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let invocation = CommandInvocation {
            name: "web_search".to_string(),
            action: CommandAction::Execute,
            arguments: json!({}),
        };
        let result = adapter
            .execute_command(&invocation)
            .await
            .expect("should succeed");
        assert!(!result.success);
        assert_eq!(result.error, Some("network error".to_string()));
    }

    #[tokio::test]
    async fn test_tool_info_to_command_stub_mapping() {
        // Direct mapping test
        let tool = ToolInfo {
            name: "my_tool".to_string(),
            description: "Does something useful".to_string(),
        };
        let client = MockToolRegistryClient::new_with_tools(vec![tool]);
        let adapter = ToolRegistryCommandAdapter::new(Arc::new(client));

        let stubs = adapter.list_commands().await.expect("should succeed");
        assert_eq!(stubs.len(), 1);
        assert_eq!(stubs[0].name, "my_tool");
        assert_eq!(stubs[0].description, "Does something useful");
    }
}
