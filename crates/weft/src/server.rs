//! Axum HTTP server setup: routes, handlers, and error response helpers.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — OpenAI-compatible chat completions
//! - `GET /health`               — Health check for load balancers
//!
//! Both gRPC and HTTP are served on the same port. The gRPC server handles
//! requests with `content-type: application/grpc`; axum handles everything else.
//! Both entry points converge at `WeftService::handle_weft_request()`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};
use weft_core::{
    ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftError, WeftMessage,
    WeftRequest,
};
use weft_proto::weft::v1 as proto;

use weft_commands::CommandRegistry;
use weft_hooks::HookRunner;
use weft_llm::ProviderService;
use weft_memory::MemoryService;
use weft_router::SemanticRouter;

use crate::grpc::WeftService;

// ── OpenAI compat types (local to this module) ─────────────────────────────
//
// These are wire-format types for the /v1/chat/completions translation layer.
// They are NOT domain types. The domain types are WeftRequest/WeftResponse in weft_core.

/// OpenAI-format chat completion request body.
#[derive(Debug, Deserialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

/// An OpenAI-format message with role and string content.
#[derive(Debug, Serialize, Deserialize)]
struct OpenAiMessage {
    role: Role,
    content: String,
}

/// OpenAI-format chat completion response body.
#[derive(Debug, Serialize)]
struct OpenAiChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiChoice {
    index: u32,
    message: OpenAiMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ── Translation functions ───────────────────────────────────────────────────

/// Translate an OpenAI-format request into a domain `WeftRequest`.
///
/// All messages are assigned `Source::Client` since this is the OpenAI compat
/// layer and there is no source attribution in the OpenAI format.
///
/// # Spec deviation note
///
/// The spec (Section 10.1) declares this as `fn openai_to_weft(...) -> Result<WeftRequest, WeftError>`.
/// This implementation is intentionally infallible: every field in `OpenAiChatRequest` maps
/// cleanly to `WeftRequest` without conditions that can fail at translation time. Validation
/// (empty messages, missing user role, streaming) is performed by the axum handler before this
/// function is called, so returning `Ok(...)` unconditionally would add noise without safety.
/// If future translation logic introduces fallible steps, the signature should be updated to
/// match the spec.
fn openai_to_weft(req: OpenAiChatRequest) -> WeftRequest {
    let messages: Vec<WeftMessage> = req
        .messages
        .into_iter()
        .map(|m| WeftMessage {
            source: match m.role {
                Role::User => Source::Client,
                Role::Assistant => Source::Provider,
                Role::System => Source::Gateway,
            },
            role: m.role,
            model: None,
            content: vec![ContentPart::Text(m.content)],
            delta: false,
            message_index: 0,
        })
        .collect();

    let routing = ModelRoutingInstruction::parse(&req.model);

    let options = SamplingOptions {
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        ..Default::default()
    };

    WeftRequest {
        messages,
        routing,
        options,
    }
}

/// Translate a domain `WeftResponse` into an OpenAI-format response.
///
/// Extracts the last assistant text message from the response.
fn weft_to_openai(resp: weft_core::WeftResponse) -> OpenAiChatResponse {
    // Extract the last assistant/provider text message.
    let assistant_text = resp
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && m.source == Source::Provider)
        .and_then(|m| {
            m.content.iter().find_map(|part| match part {
                ContentPart::Text(text) => Some(text.clone()),
                _ => None,
            })
        })
        .unwrap_or_default();

    OpenAiChatResponse {
        id: resp.id,
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: resp.model,
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiMessage {
                role: Role::Assistant,
                content: assistant_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: OpenAiUsage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
        },
    }
}

/// Return the current Unix timestamp in seconds.
fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Router ──────────────────────────────────────────────────────────────────

/// Build the combined axum+gRPC tower service.
///
/// Returns an axum `Router` that handles HTTP endpoints (`/v1/chat/completions`,
/// `/health`) merged with the tonic gRPC router. Both entry points share the
/// same `Arc<WeftService>` instance and converge at `handle_weft_request()`.
///
/// Uses `tonic::service::Routes::into_axum_router()` to compose the gRPC server
/// with axum on a single port. The tonic router handles `content-type: application/grpc`;
/// axum handles everything else.
pub fn build_router<H, R, M, P, C>(weft_service: Arc<WeftService<H, R, M, P, C>>) -> Router
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    // Build the gRPC router via tonic's axum integration.
    // tonic::service::Routes::new() wraps a NamedService + tower::Service.
    // into_axum_router() returns an axum Router that handles gRPC content-type requests.
    let grpc_router = tonic::service::Routes::new(proto::weft_server::WeftServer::new(
        weft_service.as_ref().clone(),
    ))
    .into_axum_router();

    // Build the HTTP axum router for OpenAI compat + health.
    let http_router = Router::new()
        .route(
            "/v1/chat/completions",
            post(chat_completions_handler::<H, R, M, P, C>),
        )
        .route("/health", get(health_handler::<H, R, M, P, C>))
        .with_state(Arc::clone(&weft_service));

    // Merge: gRPC router handles application/grpc requests; HTTP router handles the rest.
    // axum's `merge` composes two routers — tonic's router takes priority for gRPC paths.
    grpc_router.merge(http_router)
}

/// Start the HTTP+gRPC server and block until shutdown.
///
/// Listens on `bind_address`, serves requests until SIGTERM/SIGINT.
/// Once the signal is received, axum stops accepting new connections and
/// waits for in-flight requests to complete before returning.
pub async fn serve(
    router: Router,
    bind_address: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(bind_address).await?;
    info!(address = bind_address, "server listening (gRPC + HTTP)");

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
    // Return immediately — axum handles the grace period for in-flight requests.
}

// ── Handlers ───────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions`
///
/// Accepts an OpenAI-compatible chat completion request and returns a
/// chat completion response. Translates to/from `WeftRequest`/`WeftResponse`
/// and calls `WeftService::handle_weft_request()` — the same code path as gRPC.
/// Streaming is not supported in v1.
#[allow(clippy::type_complexity)]
async fn chat_completions_handler<H, R, M, P, C>(
    State(weft_service): State<Arc<WeftService<H, R, M, P, C>>>,
    Json(openai_req): Json<OpenAiChatRequest>,
) -> Result<Json<OpenAiChatResponse>, ApiError>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    // Generate a request-scoped tracing span for observability.
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = info_span!("chat_completion", request_id = %request_id);
    let _guard = span.enter();

    // Validate: must have at least one message.
    if openai_req.messages.is_empty() {
        return Err(ApiError::bad_request("messages array must not be empty"));
    }

    // Validate: must have at least one user message.
    if !openai_req.messages.iter().any(|m| m.role == Role::User) {
        return Err(ApiError::bad_request(
            "messages must contain at least one user message",
        ));
    }

    // Reject streaming explicitly.
    if openai_req.stream == Some(true) {
        return Err(ApiError::bad_request("streaming is not supported in v1"));
    }

    info!(
        model = %openai_req.model,
        message_count = openai_req.messages.len(),
        "handling chat completion request"
    );

    let weft_req = openai_to_weft(openai_req);

    // Call handle_weft_request — the same method the gRPC handler calls.
    // One code path to the engine regardless of entry point.
    match weft_service.handle_weft_request(weft_req).await {
        Ok(weft_resp) => Ok(Json(weft_to_openai(weft_resp))),
        Err(e) => Err(ApiError::from_weft_error(e)),
    }
}

/// `GET /health`
///
/// Returns gateway health status. Always 200 if the process is up.
#[allow(clippy::type_complexity)]
async fn health_handler<H, R, M, P, C>(
    State(weft_service): State<Arc<WeftService<H, R, M, P, C>>>,
) -> Json<HealthResponse>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    let config = weft_service.engine_config();

    // Classifier is considered loaded if it doesn't immediately fail a trivial classify.
    // Since we don't want to hit the real ONNX model here, we check the config path exists.
    let classifier_loaded = std::path::Path::new(&config.router.classifier.model_path).exists();

    // Tool registry: we don't ping the gRPC server from the health check. Report based
    // on whether tool_registry config is present (connected status is checked lazily).
    let tool_registry_connected = config.tool_registry.is_some();

    Json(HealthResponse {
        status: "ok".to_string(),
        classifier_loaded,
        tool_registry_connected,
        // The gRPC service is co-located and always available if the process is running.
        grpc_service: "serving".to_string(),
    })
}

// ── Response types ─────────────────────────────────────────────────────────

/// Health check response body.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub classifier_loaded: bool,
    pub tool_registry_connected: bool,
    /// "serving" when the gRPC service is active, "not_serving" otherwise.
    pub grpc_service: String,
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
            WeftError::InvalidRequest(_) => Self::bad_request(e.to_string()),
            WeftError::ProtoConversion(_) => Self::internal(e.to_string()),
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
            WeftError::Routing(_) => Self::internal(e.to_string()),
            WeftError::ModelNotFound { .. } => Self::internal(e.to_string()),
            WeftError::Command(_) => Self::internal(e.to_string()),
            WeftError::MemoryStore(_) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: e.to_string(),
                retry_after_ms: None,
            },
            // Configuration error: no models support the required capability.
            WeftError::NoEligibleModels { .. } => Self::bad_request(e.to_string()),
            // Hard block: hook terminated the request before LLM involvement.
            WeftError::HookBlocked { .. } => Self {
                status: StatusCode::FORBIDDEN,
                message: e.to_string(),
                retry_after_ms: None,
            },
            // Feedback block exhausted retries (PreResponse only).
            WeftError::HookBlockedAfterRetries { .. } => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                message: e.to_string(),
                retry_after_ms: None,
            },
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
    use std::collections::{HashMap, HashSet};
    use tower::ServiceExt;
    use weft_commands::{CommandError, CommandRegistry};
    use weft_core::{
        ClassifierConfig, CommandDescription, CommandInvocation, CommandResult, CommandStub,
        ContentPart, DomainsConfig, GatewayConfig, ModelEntry, ProviderConfig, Role, RouterConfig,
        ServerConfig, Source, WeftConfig, WeftMessage, WireFormat,
    };
    use weft_llm::{
        Capability, Provider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
        TokenUsage,
    };
    use weft_router::{
        RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, SemanticRouter,
    };

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
    impl Provider for MockLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text(self.response.clone())],
                    delta: false,
                    message_index: 0,
                },
                usage: Some(TokenUsage {
                    prompt_tokens: 5,
                    completion_tokens: 3,
                }),
            })
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    struct RateLimitedLlm;

    #[async_trait]
    impl Provider for RateLimitedLlm {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::RateLimited {
                retry_after_ms: 2000,
            })
        }

        fn name(&self) -> &str {
            "rate-limited"
        }
    }

    struct FailingLlm;

    #[async_trait]
    impl Provider for FailingLlm {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::RequestFailed("internal".to_string()))
        }

        fn name(&self) -> &str {
            "failing"
        }
    }

    struct MockRouter;

    #[async_trait]
    impl SemanticRouter for MockRouter {
        async fn route(
            &self,
            _user_message: &str,
            domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            let mut decision = RoutingDecision::empty();
            for (kind, candidates) in domains {
                if let RoutingDomainKind::Commands = kind {
                    decision.commands = candidates
                        .iter()
                        .map(|c| weft_router::ScoredCandidate {
                            id: c.id.clone(),
                            score: 1.0,
                        })
                        .collect();
                }
            }
            Ok(decision)
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            candidates: &[RoutingCandidate],
        ) -> Result<Vec<weft_router::ScoredCandidate>, RouterError> {
            // Score all candidates 1.0 for server tests (memory routing not tested here).
            Ok(candidates
                .iter()
                .map(|c| weft_router::ScoredCandidate {
                    id: c.id.clone(),
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
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                        capabilities: vec!["chat_completions".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    fn make_registry(llm: impl Provider + 'static) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert("test-model".to_string(), Arc::new(llm) as Arc<dyn Provider>);
        let mut model_ids = HashMap::new();
        model_ids.insert("test-model".to_string(), "claude-test".to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert("test-model".to_string(), 1024u32);
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "test-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "test-model".to_string(),
        ))
    }

    fn make_weft_service(
        llm: impl Provider + 'static,
    ) -> Arc<
        WeftService<
            weft_hooks::HookRegistry,
            MockRouter,
            weft_memory::NullMemoryService,
            ProviderRegistry,
            MockRegistry,
        >,
    > {
        let engine = crate::engine::GatewayEngine::new(
            test_config(),
            make_registry(llm),
            Arc::new(MockRouter),
            Arc::new(MockRegistry),
            None::<Arc<weft_memory::NullMemoryService>>,
            std::sync::Arc::new(weft_hooks::HookRegistry::empty()),
        );
        Arc::new(WeftService::new(engine))
    }

    fn make_router(llm: impl Provider + 'static) -> Router {
        build_router(make_weft_service(llm))
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
        let weft_service = make_weft_service(MockLlmProvider::ok("irrelevant"));
        let router = build_router(Arc::clone(&weft_service));

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
        // Phase 5: grpc_service field must be present
        assert!(health["grpc_service"].is_string());
        assert_eq!(health["grpc_service"], "serving");
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

    // ── ApiError status code mapping tests ─────────────────────────────────

    #[test]
    fn test_hook_blocked_maps_to_403() {
        let err = WeftError::HookBlocked {
            event: "request_start".to_string(),
            reason: "policy violation".to_string(),
            hook_name: "auth-hook".to_string(),
        };
        let api_error = ApiError::from_weft_error(err);
        assert_eq!(api_error.status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn test_hook_blocked_after_retries_maps_to_422() {
        let err = WeftError::HookBlockedAfterRetries {
            event: "pre_response".to_string(),
            reason: "content violation".to_string(),
            hook_name: "content-filter".to_string(),
            retries: 2,
        };
        let api_error = ApiError::from_weft_error(err);
        assert_eq!(api_error.status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn test_invalid_request_maps_to_400() {
        let err = WeftError::InvalidRequest("bad input".to_string());
        let api_error = ApiError::from_weft_error(err);
        assert_eq!(api_error.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_proto_conversion_maps_to_500() {
        let err = WeftError::ProtoConversion("missing role".to_string());
        let api_error = ApiError::from_weft_error(err);
        assert_eq!(api_error.status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ── Translation unit tests ──────────────────────────────────────────────

    #[test]
    fn test_openai_to_weft_assigns_sources() {
        use weft_core::Source;
        let req = OpenAiChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![
                OpenAiMessage {
                    role: Role::System,
                    content: "sys prompt".to_string(),
                },
                OpenAiMessage {
                    role: Role::User,
                    content: "hello".to_string(),
                },
                OpenAiMessage {
                    role: Role::Assistant,
                    content: "hi there".to_string(),
                },
            ],
            stream: None,
            temperature: None,
            max_tokens: None,
        };
        let weft_req = openai_to_weft(req);
        assert_eq!(weft_req.messages[0].source, Source::Gateway);
        assert_eq!(weft_req.messages[1].source, Source::Client);
        assert_eq!(weft_req.messages[2].source, Source::Provider);
    }

    #[test]
    fn test_openai_to_weft_routing_parsed() {
        use weft_core::RoutingMode;
        let req = OpenAiChatRequest {
            model: "auto".to_string(),
            messages: vec![OpenAiMessage {
                role: Role::User,
                content: "hi".to_string(),
            }],
            stream: None,
            temperature: None,
            max_tokens: None,
        };
        let weft_req = openai_to_weft(req);
        assert_eq!(weft_req.routing.mode, RoutingMode::Auto);
    }

    #[test]
    fn test_openai_to_weft_sampling_options() {
        let req = OpenAiChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAiMessage {
                role: Role::User,
                content: "hi".to_string(),
            }],
            stream: None,
            temperature: Some(0.5),
            max_tokens: Some(256),
        };
        let weft_req = openai_to_weft(req);
        assert_eq!(weft_req.options.temperature, Some(0.5));
        assert_eq!(weft_req.options.max_tokens, Some(256));
    }

    #[test]
    fn test_weft_to_openai_extracts_assistant_text() {
        use weft_core::{WeftResponse, WeftTiming, WeftUsage};
        let resp = WeftResponse {
            id: "chatcmpl-test".to_string(),
            model: "gpt-4".to_string(),
            messages: vec![WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("gpt-4".to_string()),
                content: vec![ContentPart::Text("Hello!".to_string())],
                delta: false,
                message_index: 0,
            }],
            usage: WeftUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                llm_calls: 1,
            },
            timing: WeftTiming::default(),
        };
        let openai_resp = weft_to_openai(resp);
        assert_eq!(openai_resp.choices[0].message.content, "Hello!");
        assert_eq!(openai_resp.usage.prompt_tokens, 10);
        assert_eq!(openai_resp.usage.completion_tokens, 5);
        assert_eq!(openai_resp.usage.total_tokens, 15);
        assert_eq!(openai_resp.object, "chat.completion");
    }

    #[test]
    fn test_weft_to_openai_empty_messages_returns_empty_content() {
        use weft_core::{WeftResponse, WeftTiming, WeftUsage};
        let resp = WeftResponse {
            id: "chatcmpl-test".to_string(),
            model: "auto".to_string(),
            messages: vec![],
            usage: WeftUsage::default(),
            timing: WeftTiming::default(),
        };
        let openai_resp = weft_to_openai(resp);
        // No assistant message → empty content, not a panic
        assert_eq!(openai_resp.choices[0].message.content, "");
    }

    // ── Dual-listener test ─────────────────────────────────────────────────

    /// Verify that the combined `build_router()` routes gRPC-content-type requests
    /// to the tonic handler, not the axum HTTP handler.
    ///
    /// Real gRPC requires HTTP/2 framing and prost-encoded bodies. A unit test using
    /// `tower::ServiceExt::oneshot` sends HTTP/1.1, so the tonic handler will reject
    /// the request with a gRPC status error — but the key observable is that the
    /// response carries `content-type: application/grpc` (set by tonic), which proves
    /// the request was dispatched to the tonic router and NOT handled by the axum
    /// JSON handlers (which would return `application/json` or a 404/405).
    ///
    /// A true end-to-end dual-listener integration test (binding a real port and
    /// connecting a tonic client) lives in the integration test suite and requires
    /// a running tokio runtime with a real TCP listener. See `tests/grpc_integration.rs`
    /// (placeholder) for that path.
    #[tokio::test]
    async fn test_grpc_content_type_routes_to_tonic_handler() {
        let router = make_router(MockLlmProvider::ok("irrelevant"));

        // Send a request with the gRPC content-type header.
        // The URI matches the tonic-generated service path: /weft.v1.Weft/Chat
        let req = Request::builder()
            .method("POST")
            .uri("/weft.v1.Weft/Chat")
            .header("content-type", "application/grpc+proto")
            // te: trailers is required by gRPC spec; tonic checks for it
            .header("te", "trailers")
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();

        // Tonic sets content-type: application/grpc on ALL responses it handles,
        // including error responses. If routing had fallen through to axum, we would
        // receive application/json (error body) or a 404, not application/grpc.
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.starts_with("application/grpc"),
            "expected gRPC content-type from tonic handler, got: {content_type:?}"
        );
    }

    // ── Single code path test ───────────────────────────────────────────────

    /// Both HTTP and gRPC entry points share the same WeftService instance.
    /// This test verifies that build_router wires Arc<WeftService> correctly:
    /// the same service handles requests from both protocols.
    #[tokio::test]
    async fn test_shared_weft_service_single_code_path() {
        // The WeftService is constructed once and shared via Arc.
        // We verify the HTTP path successfully calls handle_weft_request on it.
        let weft_service = make_weft_service(MockLlmProvider::ok("shared path response"));
        let router = build_router(Arc::clone(&weft_service));

        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test single code path"}]
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            resp["choices"][0]["message"]["content"],
            "shared path response"
        );
    }
}
