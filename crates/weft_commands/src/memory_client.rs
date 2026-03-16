//! gRPC MemoryStore client implementation via tonic.
//!
//! `MemoryStoreClient` is the trait; `GrpcMemoryStoreClient` is the gRPC implementation.
//! Connection is lazy: established on first use, cached via `tokio::sync::OnceCell`.
//!
//! Follows the same pattern as `GrpcToolRegistryClient`.

use async_trait::async_trait;
use tokio::sync::OnceCell;
use tracing::{debug, warn};

use crate::memory_types::{MemoryEntry, MemoryStoreResult};

// Include the tonic-generated code.
pub(crate) mod proto {
    tonic::include_proto!("weft.memory_store.v1");
}

use proto::memory_store_client::MemoryStoreClient as TonicMemoryStoreClient;

/// Client interface to a remote gRPC MemoryStore service.
///
/// Follows the same pattern as `ToolRegistryClient`.
#[async_trait]
pub trait MemoryStoreClient: Send + Sync + 'static {
    /// Query for relevant memories.
    async fn query(
        &self,
        query: &str,
        max_results: u32,
        min_score: f32,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError>;

    /// Store a new memory.
    ///
    /// `metadata` is serialized to a JSON string at the gRPC boundary.
    /// `None` maps to an empty string.
    async fn store(
        &self,
        content: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Result<MemoryStoreResult, MemoryStoreError>;
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryStoreError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("query failed: {0}")]
    QueryFailed(String),
    #[error("store failed: {0}")]
    StoreFailed(String),
    #[error("grpc error: {0}")]
    GrpcError(String),
}

/// gRPC client for a single MemoryStore service.
///
/// Connection is lazy: established on first use, cached thereafter.
/// Thread-safe: `OnceCell` handles concurrent initialization.
pub struct GrpcMemoryStoreClient {
    endpoint: String,
    connect_timeout_ms: u64,
    /// Stored for forward compatibility — not yet enforced per-request.
    #[allow(dead_code)]
    request_timeout_ms: u64,
    // OnceCell ensures lazy initialization without a mutex on the hot path.
    channel: OnceCell<tonic::transport::Channel>,
}

impl GrpcMemoryStoreClient {
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
    async fn channel(&self) -> Result<tonic::transport::Channel, MemoryStoreError> {
        self.channel
            .get_or_try_init(|| async {
                debug!(endpoint = %self.endpoint, "connecting to memory store");
                let connect_timeout = std::time::Duration::from_millis(self.connect_timeout_ms);
                let channel = tonic::transport::Endpoint::new(self.endpoint.clone())
                    .map_err(|e| MemoryStoreError::ConnectionFailed(e.to_string()))?
                    .connect_timeout(connect_timeout)
                    .connect()
                    .await
                    .map_err(|e| {
                        warn!(
                            endpoint = %self.endpoint,
                            error = %e,
                            "memory store connection failed"
                        );
                        MemoryStoreError::ConnectionFailed(e.to_string())
                    })?;
                Ok(channel)
            })
            .await
            .cloned()
    }

    /// Build a tonic client from the cached channel.
    async fn client(
        &self,
    ) -> Result<TonicMemoryStoreClient<tonic::transport::Channel>, MemoryStoreError> {
        let channel = self.channel().await?;
        Ok(TonicMemoryStoreClient::new(channel))
    }
}

/// Map a proto `MemoryEntry` to the domain type.
///
/// Invalid `metadata_json` is logged as a warning and mapped to `None` rather
/// than failing the entire query — content and score are still valid.
fn map_proto_entry(proto_entry: proto::MemoryEntry, store_name_for_log: &str) -> MemoryEntry {
    let metadata = if proto_entry.metadata_json.is_empty() {
        None
    } else {
        match serde_json::from_str(&proto_entry.metadata_json) {
            Ok(value) => Some(value),
            Err(e) => {
                warn!(
                    store = store_name_for_log,
                    error = %e,
                    "memory store returned invalid metadata JSON — treating as None"
                );
                None
            }
        }
    };
    MemoryEntry {
        id: proto_entry.id,
        content: proto_entry.content,
        score: proto_entry.score,
        created_at: proto_entry.created_at,
        metadata,
    }
}

#[async_trait]
impl MemoryStoreClient for GrpcMemoryStoreClient {
    async fn query(
        &self,
        query: &str,
        max_results: u32,
        min_score: f32,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let mut client = self.client().await?;

        let response = client
            .query(proto::MemoryQueryRequest {
                query: query.to_string(),
                max_results,
                min_score,
            })
            .await
            .map_err(|e| MemoryStoreError::QueryFailed(e.to_string()))?;

        let entries = response
            .into_inner()
            .memories
            .into_iter()
            .map(|e| map_proto_entry(e, &self.endpoint))
            .collect();

        Ok(entries)
    }

    async fn store(
        &self,
        content: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Result<MemoryStoreResult, MemoryStoreError> {
        let mut client = self.client().await?;

        // Serialize metadata to JSON string; None maps to empty string.
        let metadata_json = match metadata {
            Some(v) => serde_json::to_string(v).map_err(|e| {
                MemoryStoreError::StoreFailed(format!("metadata serialization failed: {e}"))
            })?,
            None => String::new(),
        };

        let response = client
            .store(proto::MemoryStoreRequest {
                content: content.to_string(),
                metadata_json,
            })
            .await
            .map_err(|e| MemoryStoreError::StoreFailed(e.to_string()))?;

        Ok(MemoryStoreResult {
            id: response.into_inner().id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::memory_store_server::{MemoryStore, MemoryStoreServer};
    use std::net::SocketAddr;
    use tokio::net::TcpListener;
    use tonic::{Request, Response, Status};

    /// In-process mock MemoryStore server for testing the gRPC client.
    struct MockMemoryStoreServer {
        query_response: Option<proto::MemoryQueryResponse>,
        query_error: Option<tonic::Code>,
        store_response: Option<proto::MemoryStoreResponse>,
        store_error: Option<tonic::Code>,
    }

    impl MockMemoryStoreServer {
        fn new() -> Self {
            Self {
                query_response: None,
                query_error: None,
                store_response: None,
                store_error: None,
            }
        }

        fn with_query_response(mut self, resp: proto::MemoryQueryResponse) -> Self {
            self.query_response = Some(resp);
            self
        }

        fn with_query_error(mut self, code: tonic::Code) -> Self {
            self.query_error = Some(code);
            self
        }

        fn with_store_response(mut self, resp: proto::MemoryStoreResponse) -> Self {
            self.store_response = Some(resp);
            self
        }

        fn with_store_error(mut self, code: tonic::Code) -> Self {
            self.store_error = Some(code);
            self
        }
    }

    #[async_trait::async_trait]
    impl MemoryStore for MockMemoryStoreServer {
        async fn query(
            &self,
            _request: Request<proto::MemoryQueryRequest>,
        ) -> Result<Response<proto::MemoryQueryResponse>, Status> {
            if let Some(code) = self.query_error {
                return Err(Status::new(code, "mock query error"));
            }
            match &self.query_response {
                Some(resp) => Ok(Response::new(resp.clone())),
                None => Err(Status::internal("no mock query response configured")),
            }
        }

        async fn store(
            &self,
            _request: Request<proto::MemoryStoreRequest>,
        ) -> Result<Response<proto::MemoryStoreResponse>, Status> {
            if let Some(code) = self.store_error {
                return Err(Status::new(code, "mock store error"));
            }
            match &self.store_response {
                Some(resp) => Ok(Response::new(resp.clone())),
                None => Err(Status::internal("no mock store response configured")),
            }
        }
    }

    /// Spawn a mock server on a random port.
    async fn spawn_mock_server(
        server: MockMemoryStoreServer,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind to random port");
        let addr = listener.local_addr().expect("get local addr");

        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MemoryStoreServer::new(server))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .ok();
        });

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (addr, handle)
    }

    fn make_client(addr: SocketAddr) -> GrpcMemoryStoreClient {
        GrpcMemoryStoreClient::new(format!("http://{addr}"), 5_000, 10_000)
    }

    // ── query ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_query_maps_proto_to_domain() {
        let mock = MockMemoryStoreServer::new().with_query_response(proto::MemoryQueryResponse {
            memories: vec![
                proto::MemoryEntry {
                    id: "mem-1".to_string(),
                    content: "User prefers dark mode".to_string(),
                    score: 0.87,
                    created_at: "2026-03-15T10:00:00Z".to_string(),
                    metadata_json: String::new(),
                },
                proto::MemoryEntry {
                    id: "mem-2".to_string(),
                    content: "Timezone: America/Vancouver".to_string(),
                    score: 0.72,
                    created_at: "2026-03-15T09:00:00Z".to_string(),
                    metadata_json: r#"{"source":"conversation"}"#.to_string(),
                },
            ],
        });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let entries = client.query("user preferences", 5, 0.0).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "mem-1");
        assert_eq!(entries[0].content, "User prefers dark mode");
        assert!((entries[0].score - 0.87).abs() < 0.001);
        assert!(
            entries[0].metadata.is_none(),
            "empty metadata_json maps to None"
        );

        assert_eq!(entries[1].id, "mem-2");
        assert!(entries[1].metadata.is_some(), "valid JSON maps to Some");
    }

    #[tokio::test]
    async fn test_query_empty_metadata_maps_to_none() {
        let mock = MockMemoryStoreServer::new().with_query_response(proto::MemoryQueryResponse {
            memories: vec![proto::MemoryEntry {
                id: "m1".to_string(),
                content: "content".to_string(),
                score: 0.5,
                created_at: "2026-01-01T00:00:00Z".to_string(),
                metadata_json: String::new(),
            }],
        });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let entries = client.query("query", 5, 0.0).await.unwrap();
        assert!(entries[0].metadata.is_none());
    }

    #[tokio::test]
    async fn test_query_invalid_metadata_json_maps_to_none() {
        // Invalid JSON should warn and map to None, not fail the query.
        let mock = MockMemoryStoreServer::new().with_query_response(proto::MemoryQueryResponse {
            memories: vec![proto::MemoryEntry {
                id: "m1".to_string(),
                content: "content".to_string(),
                score: 0.5,
                created_at: "2026-01-01T00:00:00Z".to_string(),
                metadata_json: "not valid json {".to_string(),
            }],
        });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let entries = client.query("query", 5, 0.0).await.unwrap();
        assert_eq!(
            entries.len(),
            1,
            "invalid metadata should not drop the entry"
        );
        assert!(
            entries[0].metadata.is_none(),
            "invalid metadata maps to None"
        );
        assert_eq!(entries[0].content, "content", "content still valid");
    }

    #[tokio::test]
    async fn test_query_grpc_error_maps_to_error() {
        let mock = MockMemoryStoreServer::new().with_query_error(tonic::Code::Internal);

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client.query("query", 5, 0.0).await;
        assert!(
            matches!(result, Err(MemoryStoreError::QueryFailed(_))),
            "expected QueryFailed, got: {result:?}"
        );
    }

    // ── store ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_store_maps_proto_to_domain() {
        let mock = MockMemoryStoreServer::new().with_store_response(proto::MemoryStoreResponse {
            id: "mem-assigned-99".to_string(),
        });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client
            .store("the user prefers dark mode", None)
            .await
            .unwrap();
        assert_eq!(result.id, "mem-assigned-99");
    }

    #[tokio::test]
    async fn test_store_with_metadata() {
        let mock = MockMemoryStoreServer::new().with_store_response(proto::MemoryStoreResponse {
            id: "mem-100".to_string(),
        });

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let metadata = serde_json::json!({"session": "abc123"});
        let result = client.store("content", Some(&metadata)).await.unwrap();
        assert_eq!(result.id, "mem-100");
    }

    #[tokio::test]
    async fn test_store_grpc_error_maps_to_error() {
        let mock = MockMemoryStoreServer::new().with_store_error(tonic::Code::Unavailable);

        let (addr, _handle) = spawn_mock_server(mock).await;
        let client = make_client(addr);

        let result = client.store("content", None).await;
        assert!(
            matches!(result, Err(MemoryStoreError::StoreFailed(_))),
            "expected StoreFailed, got: {result:?}"
        );
    }

    // ── connection failure ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_connection_failure_returns_error() {
        let client = GrpcMemoryStoreClient::new("http://127.0.0.1:1".to_string(), 500, 1_000);

        let result = client.query("query", 5, 0.0).await;
        assert!(
            matches!(result, Err(MemoryStoreError::ConnectionFailed(_))),
            "expected ConnectionFailed, got: {result:?}"
        );
    }
}
