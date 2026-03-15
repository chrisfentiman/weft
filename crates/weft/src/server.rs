//! Axum HTTP server setup: routes, handlers, and error response helpers.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — OpenAI-compatible chat completions
//! - `GET /health`               — Health check for load balancers

use std::time::Duration;

use axum::{
    Json, Router,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};
use weft_core::{ChatCompletionRequest, ChatCompletionResponse, WeftError};

use crate::engine::GatewayEngine;

/// How long in-flight requests have to complete after shutdown signal.
const SHUTDOWN_GRACE_PERIOD: Duration = Duration::from_secs(30);

/// Build the axum `Router` with all routes attached.
pub fn build_router(engine: GatewayEngine) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/health", get(health_handler))
        .with_state(engine)
}

/// Start the HTTP server and block until shutdown.
///
/// Listens on `bind_address`, serves requests until SIGTERM/SIGINT, then
/// gives in-flight requests up to `SHUTDOWN_GRACE_PERIOD` to complete.
pub async fn serve(
    router: Router,
    bind_address: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(bind_address).await?;
    info!(address = bind_address, "server listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Wait for SIGINT or SIGTERM, then return so axum can start graceful shutdown.
async fn shutdown_signal() {
    // SIGINT (Ctrl-C) is portable; SIGTERM is Unix-only.
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("received SIGINT, beginning graceful shutdown");
            }
            _ = sigterm.recv() => {
                info!("received SIGTERM, beginning graceful shutdown");
            }
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
        info!("received Ctrl-C, beginning graceful shutdown");
    }

    // Give in-flight requests time to complete.
    tokio::time::sleep(SHUTDOWN_GRACE_PERIOD).await;
}

// ── Handlers ───────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions`
///
/// Accepts an OpenAI-compatible chat completion request and returns a
/// chat completion response. Streaming is not supported in v1.
async fn chat_completions_handler(
    State(engine): State<GatewayEngine>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    // Generate a request-scoped tracing span for observability.
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = info_span!("chat_completion", request_id = %request_id);
    let _guard = span.enter();

    // Validate: must have at least one message.
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages array must not be empty"));
    }

    // Validate: must have at least one user message.
    if !request
        .messages
        .iter()
        .any(|m| m.role == weft_core::Role::User)
    {
        return Err(ApiError::bad_request(
            "messages must contain at least one user message",
        ));
    }

    // Reject streaming explicitly.
    if request.stream == Some(true) {
        return Err(ApiError::bad_request("streaming is not supported in v1"));
    }

    info!(
        model = %request.model,
        message_count = request.messages.len(),
        "handling chat completion request"
    );

    match engine.handle_request(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err(ApiError::from_weft_error(e)),
    }
}

/// `GET /health`
///
/// Returns gateway health status. Always 200 if the process is up.
async fn health_handler(State(engine): State<GatewayEngine>) -> Json<HealthResponse> {
    // Classifier is considered loaded if it doesn't immediately fail a trivial classify.
    // Since we don't want to hit the real ONNX model here, we check the config path exists.
    let classifier_loaded = std::path::Path::new(&engine.config().classifier.model_path).exists();

    // Tool registry: we don't ping the gRPC server from the health check. Report based
    // on whether tool_registry config is present (connected status is checked lazily).
    let tool_registry_connected = engine.config().tool_registry.is_some();

    Json(HealthResponse {
        status: "ok".to_string(),
        classifier_loaded,
        tool_registry_connected,
    })
}

// ── Response types ─────────────────────────────────────────────────────────

/// Health check response body.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub classifier_loaded: bool,
    pub tool_registry_connected: bool,
}

/// OpenAI-compatible error response body.
#[derive(Debug, Serialize, Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize, Deserialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

/// API error that converts to an HTTP response with the right status code.
pub struct ApiError {
    status: StatusCode,
    message: String,
    /// For rate-limit errors: the retry-after value in milliseconds.
    retry_after_ms: Option<u64>,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            retry_after_ms: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            retry_after_ms: None,
        }
    }

    fn from_weft_error(e: WeftError) -> Self {
        match e {
            WeftError::StreamingNotSupported => Self::bad_request(e.to_string()),
            WeftError::Config(_) => Self::internal(e.to_string()),
            WeftError::RateLimited { retry_after_ms } => Self {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: e.to_string(),
                retry_after_ms: Some(retry_after_ms),
            },
            WeftError::RequestTimeout { .. } => Self {
                status: StatusCode::GATEWAY_TIMEOUT,
                message: e.to_string(),
                retry_after_ms: None,
            },
            WeftError::ToolRegistry(_) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: e.to_string(),
                retry_after_ms: None,
            },
            WeftError::CommandLoopExceeded { .. } => Self::internal(e.to_string()),
            WeftError::Llm(_) => Self::internal(e.to_string()),
            WeftError::Classifier(_) => Self::internal(e.to_string()),
            WeftError::Command(_) => Self::internal(e.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                message: self.message,
                error_type: "invalid_request_error".to_string(),
                code: None,
            },
        };

        let json_body = serde_json::to_string(&body).unwrap_or_else(|_| {
            r#"{"error":{"message":"internal serialization error","type":"internal_error","code":null}}"#
                .to_string()
        });

        let mut response = Response::builder()
            .status(self.status)
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(json_body))
            .unwrap_or_else(|_| {
                warn!("failed to build error response");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            });

        // Add Retry-After header for rate-limit responses.
        if let Some(retry_after_ms) = self.retry_after_ms {
            let retry_after_secs = retry_after_ms.div_ceil(1000);
            if let Ok(value) = header::HeaderValue::from_str(&retry_after_secs.to_string()) {
                response.headers_mut().insert(header::RETRY_AFTER, value);
            }
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use serde_json::{Value, json};
    use std::sync::Arc;
    use tower::ServiceExt;
    use weft_classifier::{ClassificationResult, ClassifierError, SemanticClassifier};
    use weft_commands::{CommandError, CommandRegistry};
    use weft_core::{
        ClassifierConfig, CommandDescription, CommandInvocation, CommandResult, CommandStub,
        GatewayConfig, LlmConfig, LlmProviderKind, Message, ServerConfig, WeftConfig,
    };
    use weft_llm::{CompletionOptions, CompletionResponse, LlmError, LlmProvider};

    // ── Test mocks (same as in engine.rs tests) ────────────────────────────

    struct MockLlmProvider {
        response: String,
    }

    impl MockLlmProvider {
        fn ok(s: &str) -> Self {
            Self {
                response: s.to_string(),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockLlmProvider {
        async fn complete(
            &self,
            _: &str,
            _: &[Message],
            _: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                text: self.response.clone(),
                usage: Some(weft_llm::LlmUsage {
                    prompt_tokens: 5,
                    completion_tokens: 3,
                }),
            })
        }
    }

    struct RateLimitedLlm;

    #[async_trait]
    impl LlmProvider for RateLimitedLlm {
        async fn complete(
            &self,
            _: &str,
            _: &[Message],
            _: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::RateLimited {
                retry_after_ms: 2000,
            })
        }
    }

    struct FailingLlm;

    #[async_trait]
    impl LlmProvider for FailingLlm {
        async fn complete(
            &self,
            _: &str,
            _: &[Message],
            _: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::RequestFailed("internal".to_string()))
        }
    }

    struct MockClassifier;

    #[async_trait]
    impl SemanticClassifier for MockClassifier {
        async fn classify(
            &self,
            _: &str,
            commands: &[CommandStub],
        ) -> Result<Vec<ClassificationResult>, ClassifierError> {
            Ok(commands
                .iter()
                .map(|c| ClassificationResult {
                    command_name: c.name.clone(),
                    score: 1.0,
                })
                .collect())
        }
    }

    struct MockRegistry;

    #[async_trait]
    impl CommandRegistry for MockRegistry {
        async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
            Ok(vec![])
        }

        async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
            Err(CommandError::NotFound(name.to_string()))
        }

        async fn execute_command(
            &self,
            invocation: &CommandInvocation,
        ) -> Result<CommandResult, CommandError> {
            Err(CommandError::NotFound(invocation.name.clone()))
        }
    }

    fn test_config() -> Arc<WeftConfig> {
        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:0".to_string(),
            },
            llm: LlmConfig {
                provider: LlmProviderKind::Anthropic,
                api_key: "test-key".to_string(),
                model: "claude-test".to_string(),
                max_tokens: 1024,
                base_url: None,
            },
            classifier: ClassifierConfig {
                model_path: "models/test.onnx".to_string(),
                tokenizer_path: "models/tokenizer.json".to_string(),
                threshold: 0.0,
                max_commands: 20,
            },
            tool_registry: None,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    fn make_router(llm: impl LlmProvider + 'static) -> Router {
        let engine = GatewayEngine::new(
            test_config(),
            Arc::new(llm),
            Arc::new(MockClassifier),
            Arc::new(MockRegistry),
        );
        build_router(engine)
    }

    async fn post_json(router: Router, body: Value) -> (StatusCode, Value) {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }

    // ── HTTP tests ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_valid_request_returns_200() {
        let router = make_router(MockLlmProvider::ok("Hello there!"));
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resp["object"], "chat.completion");
        assert_eq!(resp["choices"][0]["message"]["content"], "Hello there!");
        assert!(resp["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    #[tokio::test]
    async fn test_invalid_json_returns_4xx() {
        let router = make_router(MockLlmProvider::ok("irrelevant"));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(b"not valid json".as_ref()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        // axum returns 400 or 422 for invalid JSON depending on version; either is correct
        assert!(
            resp.status().is_client_error(),
            "expected 4xx for invalid JSON, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_empty_messages_returns_400() {
        let router = make_router(MockLlmProvider::ok("irrelevant"));
        let body = json!({
            "model": "gpt-4",
            "messages": []
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(resp["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn test_no_user_message_returns_400() {
        let router = make_router(MockLlmProvider::ok("irrelevant"));
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "system only"}]
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(resp["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn test_stream_true_returns_400() {
        let router = make_router(MockLlmProvider::ok("irrelevant"));
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": true
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(resp["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn test_llm_error_returns_500() {
        let router = make_router(FailingLlm);
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let (status, _) = post_json(router, body).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_rate_limited_returns_429_with_retry_after() {
        let router = make_router(RateLimitedLlm);
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        // retry_after_ms=2000 → Retry-After: 2 seconds
        assert_eq!(
            resp.headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok()),
            Some("2")
        );
    }

    #[tokio::test]
    async fn test_response_contains_usage() {
        let router = make_router(MockLlmProvider::ok("Done"));
        let body = json!({
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::OK);
        let usage = &resp["usage"];
        assert_eq!(usage["prompt_tokens"], 5);
        assert_eq!(usage["completion_tokens"], 3);
        assert_eq!(usage["total_tokens"], 8);
    }

    #[tokio::test]
    async fn test_health_endpoint_returns_200() {
        let engine = GatewayEngine::new(
            test_config(),
            Arc::new(MockLlmProvider::ok("irrelevant")),
            Arc::new(MockClassifier),
            Arc::new(MockRegistry),
        );
        let router = build_router(engine);

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(health["status"], "ok");
        assert!(health["classifier_loaded"].is_boolean());
        assert!(health["tool_registry_connected"].is_boolean());
    }

    #[tokio::test]
    async fn test_response_model_preserved_from_request() {
        let router = make_router(MockLlmProvider::ok("Hi"));
        let body = json!({
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(resp["model"], "my-custom-model");
    }

    #[tokio::test]
    async fn test_error_body_format() {
        let router = make_router(FailingLlm);
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let (_, resp) = post_json(router, body).await;

        // Must have the OpenAI error envelope
        assert!(resp["error"]["message"].is_string());
        assert!(resp["error"]["type"].is_string());
        // code is present (may be null)
        assert!(resp["error"].get("code").is_some());
    }
}
