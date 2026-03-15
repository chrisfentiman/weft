//! gRPC ToolRegistry client implementation via tonic.
//!
//! Uses lazy connection: connects on first use, not at startup.
//! Connection is cached after first successful connect via `tokio::sync::OnceCell`.

use async_trait::async_trait;
use tokio::sync::OnceCell;
use tracing::{debug, warn};

use crate::{
    ToolDescription, ToolExecutionResult, ToolInfo, ToolRegistryClient, ToolRegistryError,
};

// Include the tonic-generated code.
pub(crate) mod proto {
    tonic::include_proto!("weft.tool_registry.v1");
}

use proto::tool_registry_client::ToolRegistryClient as TonicClient;

/// gRPC client for the ToolRegistry service.
///
/// Connection is lazy: the channel is established on first use and reused thereafter.
pub struct GrpcToolRegistryClient {
    endpoint: String,
    connect_timeout_ms: u64,
    /// Stored for future use when a timeout layer is applied per-request.
    #[allow(dead_code)]
    request_timeout_ms: u64,
    // OnceCell ensures lazy initialization without a mutex on the hot path.
    channel: OnceCell<tonic::transport::Channel>,
}

impl GrpcToolRegistryClient {
    /// Create a new client. Does not connect immediately.
    pub fn new(endpoint: String, connect_timeout_ms: u64, request_timeout_ms: u64) -> Self {
        Self {
            endpoint,
            connect_timeout_ms,
            request_timeout_ms,
            channel: OnceCell::new(),
        }
    }

    /// Get or create the tonic channel. Lazy-initializes on first call.
    async fn channel(&self) -> Result<tonic::transport::Channel, ToolRegistryError> {
        self.channel
            .get_or_try_init(|| async {
                debug!(endpoint = %self.endpoint, "connecting to tool registry");
                let connect_timeout =
                    std::time::Duration::from_millis(self.connect_timeout_ms);
                let channel = tonic::transport::Endpoint::new(self.endpoint.clone())
                    .map_err(|e| ToolRegistryError::ConnectionFailed(e.to_string()))?
                    .connect_timeout(connect_timeout)
                    .connect()
                    .await
                    .map_err(|e| {
                        warn!(endpoint = %self.endpoint, error = %e, "tool registry connection failed");
                        ToolRegistryError::ConnectionFailed(e.to_string())
                    })?;
                Ok(channel)
            })
            .await
            .cloned()
    }

    /// Build a tonic client with the request timeout applied.
    async fn client(&self) -> Result<TonicClient<tonic::transport::Channel>, ToolRegistryError> {
        let channel = self.channel().await?;
        Ok(TonicClient::new(channel))
    }
}

#[async_trait]
impl ToolRegistryClient for GrpcToolRegistryClient {
    async fn list_tools(&self) -> Result<Vec<ToolInfo>, ToolRegistryError> {
        let mut client = self.client().await?;

        let response = client
            .list_tools(proto::ListToolsRequest {})
            .await
            .map_err(|e| ToolRegistryError::GrpcError(e.to_string()))?;

        let tools = response
            .into_inner()
            .tools
            .into_iter()
            .map(|t| ToolInfo {
                name: t.name,
                description: t.description,
            })
            .collect();

        Ok(tools)
    }

    async fn describe_tool(&self, name: &str) -> Result<ToolDescription, ToolRegistryError> {
        let mut client = self.client().await?;

        let response = client
            .describe_tool(proto::DescribeToolRequest {
                name: name.to_string(),
            })
            .await
            .map_err(|e| {
                let status = e.code();
                if status == tonic::Code::NotFound {
                    ToolRegistryError::ToolNotFound(name.to_string())
                } else {
                    ToolRegistryError::GrpcError(e.to_string())
                }
            })?;

        let r = response.into_inner();
        let parameters_schema =
            if r.parameters_schema.is_empty() {
                None
            } else {
                Some(serde_json::from_str(&r.parameters_schema).map_err(|e| {
                    ToolRegistryError::GrpcError(format!("invalid schema JSON: {e}"))
                })?)
            };

        Ok(ToolDescription {
            name: r.name,
            description: r.description,
            usage: r.usage,
            parameters_schema,
        })
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecutionResult, ToolRegistryError> {
        let mut client = self.client().await?;

        let arguments_json = serde_json::to_string(&arguments).map_err(|e| {
            ToolRegistryError::GrpcError(format!("argument serialization failed: {e}"))
        })?;

        let response = client
            .execute_tool(proto::ExecuteToolRequest {
                name: name.to_string(),
                arguments_json,
            })
            .await
            .map_err(|e| {
                let status = e.code();
                if status == tonic::Code::NotFound {
                    ToolRegistryError::ToolNotFound(name.to_string())
                } else {
                    ToolRegistryError::ExecutionFailed(e.to_string())
                }
            })?;

        let r = response.into_inner();
        let error = if r.error.is_empty() {
            None
        } else {
            Some(r.error)
        };

        Ok(ToolExecutionResult {
            success: r.success,
            output: r.output,
            error,
        })
    }
}
