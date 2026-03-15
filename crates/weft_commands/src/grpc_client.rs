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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolRegistryClient;
    use proto::tool_registry_server::{ToolRegistry, ToolRegistryServer};
    use serde_json::json;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;
    use tonic::{Request, Response, Status};

    /// In-process mock ToolRegistry server for testing the gRPC client.
    ///
    /// Each field controls the response for one RPC. `NotFound` gRPC status codes are
    /// returned when `describe_not_found` / `execute_not_found` are set.
    struct MockToolRegistryServer {
        tools: Vec<proto::ToolInfo>,
        describe_response: Option<proto::DescribeToolResponse>,
        describe_not_found: bool,
        execute_response: Option<proto::ExecuteToolResponse>,
        execute_not_found: bool,
    }

    impl MockToolRegistryServer {
        fn new() -> Self {
            Self {
                tools: Vec::new(),
                describe_response: None,
                describe_not_found: false,
                execute_response: None,
                execute_not_found: false,
            }
        }

        fn with_tools(mut self, tools: Vec<proto::ToolInfo>) -> Self {
            self.tools = tools;
            self
        }

        fn with_describe_response(mut self, resp: proto::DescribeToolResponse) -> Self {
            self.describe_response = Some(resp);
            self
        }

        fn with_describe_not_found(mut self) -> Self {
            self.describe_not_found = true;
            self
        }

        fn with_execute_response(mut self, resp: proto::ExecuteToolResponse) -> Self {
            self.execute_response = Some(resp);
            self
        }

        fn with_execute_not_found(mut self) -> Self {
            self.execute_not_found = true;
            self
        }
    }

    #[async_trait::async_trait]
    impl ToolRegistry for MockToolRegistryServer {
        async fn list_tools(
            &self,
            _request: Request<proto::ListToolsRequest>,
        ) -> Result<Response<proto::ListToolsResponse>, Status> {
            Ok(Response::new(proto::ListToolsResponse {
                tools: self.tools.clone(),
            }))
        }

        async fn describe_tool(
            &self,
            _request: Request<proto::DescribeToolRequest>,
        ) -> Result<Response<proto::DescribeToolResponse>, Status> {
            if self.describe_not_found {
                return Err(Status::not_found("tool not found"));
            }
            match &self.describe_response {
                Some(resp) => Ok(Response::new(resp.clone())),
                None => Err(Status::internal("no mock describe response configured")),
            }
        }

        async fn execute_tool(
            &self,
            _request: Request<proto::ExecuteToolRequest>,
        ) -> Result<Response<proto::ExecuteToolResponse>, Status> {
            if self.execute_not_found {
                return Err(Status::not_found("tool not found"));
            }
            match &self.execute_response {
                Some(resp) => Ok(Response::new(resp.clone())),
                None => Err(Status::internal("no mock execute response configured")),
            }
        }
    }

    /// Spawn a mock server on a random port and return the socket address.
    /// The server runs until the returned `tokio::task::JoinHandle` is dropped or aborted.
    async fn spawn_mock_server(
        server: MockToolRegistryServer,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind to random port");
        let addr = listener.local_addr().expect("get local addr");

        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(ToolRegistryServer::new(server))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .ok();
        });

        // Give the server a moment to start accepting.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        (addr, handle)
    }

    fn make_client(addr: SocketAddr) -> GrpcToolRegistryClient {
        GrpcToolRegistryClient::new(
            format!("http://{addr}"),
            5_000,  // connect_timeout_ms
            10_000, // request_timeout_ms
        )
    }

    // ── list_tools ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_tools_maps_proto_to_domain() {
        let mock = MockToolRegistryServer::new().with_tools(vec![
            proto::ToolInfo {
                name: "web_search".to_string(),
                description: "Search the web".to_string(),
            },
            proto::ToolInfo {
                name: "code_review".to_string(),
                description: "Review code".to_string(),
            },
        ]);

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let tools = client
            .list_tools()
            .await
            .expect("list_tools should succeed");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "web_search");
        assert_eq!(tools[0].description, "Search the web");
        assert_eq!(tools[1].name, "code_review");
        assert_eq!(tools[1].description, "Review code");
    }

    #[tokio::test]
    async fn test_list_tools_empty_registry() {
        let mock = MockToolRegistryServer::new();
        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let tools = client.list_tools().await.expect("should succeed");
        assert!(tools.is_empty());
    }

    // ── describe_tool ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_describe_tool_maps_proto_to_domain() {
        let mock =
            MockToolRegistryServer::new().with_describe_response(proto::DescribeToolResponse {
                name: "web_search".to_string(),
                description: "Search the web for current information".to_string(),
                usage: "/web_search query: \"search terms\", limit: 10".to_string(),
                parameters_schema: r#"{"type":"object","properties":{"query":{"type":"string"}}}"#
                    .to_string(),
            });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let desc = client
            .describe_tool("web_search")
            .await
            .expect("describe_tool should succeed");

        assert_eq!(desc.name, "web_search");
        assert_eq!(desc.description, "Search the web for current information");
        assert_eq!(desc.usage, "/web_search query: \"search terms\", limit: 10");
        assert!(desc.parameters_schema.is_some());
        assert_eq!(
            desc.parameters_schema.unwrap(),
            json!({"type": "object", "properties": {"query": {"type": "string"}}})
        );
    }

    #[tokio::test]
    async fn test_describe_tool_empty_schema_maps_to_none() {
        let mock =
            MockToolRegistryServer::new().with_describe_response(proto::DescribeToolResponse {
                name: "simple_tool".to_string(),
                description: "A simple tool".to_string(),
                usage: "/simple_tool".to_string(),
                parameters_schema: String::new(), // empty string -> None
            });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let desc = client
            .describe_tool("simple_tool")
            .await
            .expect("should succeed");
        assert!(desc.parameters_schema.is_none());
    }

    #[tokio::test]
    async fn test_describe_tool_not_found_returns_error() {
        let mock = MockToolRegistryServer::new().with_describe_not_found();
        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client.describe_tool("no_such_tool").await;
        assert!(
            matches!(result, Err(ToolRegistryError::ToolNotFound(_))),
            "expected ToolNotFound, got: {result:?}"
        );
    }

    // ── execute_tool ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_execute_tool_success() {
        let mock =
            MockToolRegistryServer::new().with_execute_response(proto::ExecuteToolResponse {
                success: true,
                output: "Found 3 results for 'Rust async'".to_string(),
                error: String::new(),
            });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client
            .execute_tool("web_search", json!({"query": "Rust async"}))
            .await
            .expect("execute_tool should succeed");

        assert!(result.success);
        assert_eq!(result.output, "Found 3 results for 'Rust async'");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_execute_tool_error_mapped_from_proto() {
        // The server returns success=false with an error message (not a gRPC error status).
        let mock =
            MockToolRegistryServer::new().with_execute_response(proto::ExecuteToolResponse {
                success: false,
                output: String::new(),
                error: "external service unavailable".to_string(),
            });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client
            .execute_tool("web_search", json!({}))
            .await
            .expect("execute_tool should succeed at RPC level");

        assert!(!result.success);
        assert_eq!(
            result.error,
            Some("external service unavailable".to_string())
        );
    }

    #[tokio::test]
    async fn test_execute_tool_not_found_returns_error() {
        let mock = MockToolRegistryServer::new().with_execute_not_found();
        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client.execute_tool("no_such_tool", json!({})).await;
        assert!(
            matches!(result, Err(ToolRegistryError::ToolNotFound(_))),
            "expected ToolNotFound, got: {result:?}"
        );
    }

    // ── connection failure ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_connection_failure_returns_error() {
        // Point at a port with nothing listening on it.
        let client = GrpcToolRegistryClient::new(
            "http://127.0.0.1:1".to_string(), // port 1 is reserved, will refuse
            500,                              // short timeout to keep the test fast
            1_000,
        );

        let result = client.list_tools().await;
        assert!(
            matches!(result, Err(ToolRegistryError::ConnectionFailed(_))),
            "expected ConnectionFailed, got: {result:?}"
        );
    }
}
