//! Axum HTTP server setup: routes, handlers, and error response helpers.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — OpenAI-compatible chat completions
//! - `GET /health`               — Health check for load balancers
//! - `GET /metrics`              — Prometheus metrics (when enabled)
//!
//! Both gRPC and HTTP are served on the same port. The gRPC server handles
//! requests with `content-type: application/grpc`; axum handles everything else.
//! Both entry points converge at `WeftService::handle_weft_request()`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{self, HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use weft_core::{
    ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftError, WeftMessage,
    WeftRequest,
};
use weft_proto::weft::v1 as proto;

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
/// Intentionally infallible: every field in `OpenAiChatRequest` maps cleanly to
/// `WeftRequest` without conditions that can fail at translation time. Validation
/// (empty messages, missing user role, streaming) is performed by the axum handler
/// before this function is called.
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
/// `request_model` is the model string from the original OpenAI request and is
/// echoed back verbatim so clients see the model they asked for, not the
/// internal routing name. The response id is prefixed with `chatcmpl-` to
/// match the OpenAI wire format.
fn weft_to_openai(resp: weft_core::WeftResponse, request_model: String) -> OpenAiChatResponse {
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

    // Prefix id with "chatcmpl-" to match OpenAI wire format.
    let id = if resp.id.starts_with("chatcmpl-") {
        resp.id
    } else {
        format!("chatcmpl-{}", resp.id)
    };

    OpenAiChatResponse {
        id,
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: request_model,
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
/// `/health`, `/metrics`) merged with the tonic gRPC router. Both entry points
/// share the same `Arc<WeftService>` instance and converge at `handle_weft_request()`.
///
/// Uses `tonic::service::Routes::into_axum_router()` to compose the gRPC server
/// with axum on a single port. The tonic router handles `content-type: application/grpc`;
/// axum handles everything else.
///
/// When `prometheus_handle` is `Some`, a `GET /metrics` route is added that renders
/// metrics in Prometheus text exposition format. When `None`, the route is omitted.
pub fn build_router(
    weft_service: Arc<WeftService>,
    prometheus_handle: Option<PrometheusHandle>,
) -> Router {
    // Build the gRPC router via tonic's axum integration.
    // tonic::service::Routes::new() wraps a NamedService + tower::Service.
    // into_axum_router() returns an axum Router that handles gRPC content-type requests.
    let grpc_router = tonic::service::Routes::new(proto::weft_server::WeftServer::new(
        weft_service.as_ref().clone(),
    ))
    .into_axum_router();

    // Build the HTTP axum router for OpenAI compat + health.
    let mut http_router = Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/health", get(health_handler))
        .with_state(Arc::clone(&weft_service));

    // Add the Prometheus /metrics endpoint when metrics are enabled.
    // PrometheusHandle is Clone, so we move it into the handler via closure state.
    if let Some(handle) = prometheus_handle {
        http_router =
            http_router.route("/metrics", get(move || prometheus_handler(handle.clone())));
    }

    // Merge: gRPC router handles application/grpc requests; HTTP router handles the rest.
    // axum's `merge` composes two routers — tonic's router takes priority for gRPC paths.
    // TraceLayer adds transport-level spans wrapping all request handling.
    grpc_router
        .merge(http_router)
        .layer(TraceLayer::new_for_http())
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
///
/// Extracts W3C TraceContext from the `traceparent` request header. When present,
/// the reactor's root `request` span becomes a child of the incoming trace, enabling
/// distributed tracing across services.
async fn chat_completions_handler(
    State(weft_service): State<Arc<WeftService>>,
    headers: HeaderMap,
    Json(openai_req): Json<OpenAiChatRequest>,
) -> Result<Json<OpenAiChatResponse>, ApiError> {
    // Extract W3C TraceContext from incoming headers and set it on the current span.
    // tower-http's TraceLayer has already created a span for this request — by setting
    // its OTel parent here, child spans inherit the incoming distributed trace.
    // `set_parent` does not use thread-local storage and is safe across await points.
    {
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let parent_ctx = crate::telemetry::propagation::extract_from_headers(&headers);
        tracing::Span::current().set_parent(parent_ctx);
    }

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

    // Capture request model before consuming openai_req (for echo in response).
    let request_model = openai_req.model.clone();

    let weft_req = openai_to_weft(openai_req);

    // Call handle_weft_request — the same method the gRPC handler calls.
    // One code path to the engine regardless of entry point.
    match weft_service.handle_weft_request(weft_req).await {
        Ok(weft_resp) => Ok(Json(weft_to_openai(weft_resp, request_model))),
        Err(e) => Err(ApiError::from_weft_error(e)),
    }
}

/// `GET /metrics`
///
/// Returns current metrics in Prometheus text exposition format.
///
/// Renders all registered metrics using the provided `PrometheusHandle`.
/// The content type is `text/plain; charset=utf-8` as required by the
/// Prometheus exposition format specification.
///
/// This endpoint has no authentication. It is on the same port as the API.
async fn prometheus_handler(handle: PrometheusHandle) -> impl IntoResponse {
    let metrics_text = handle.render();
    (
        [(
            http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics_text,
    )
}

/// `GET /health`
///
/// Returns gateway health status. Always 200 if the process is up.
async fn health_handler(State(weft_service): State<Arc<WeftService>>) -> Json<HealthResponse> {
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
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use serde_json::{Value, json};
    use std::collections::{HashMap, HashSet};
    use tower::ServiceExt;
    use weft_activities::{
        AssembleResponseActivity, CommandFormattingActivity, CommandSelectionActivity,
        ExecuteCommandActivity, GenerateActivity, HookActivity, ModelSelectionActivity,
        ProviderResolutionActivity, SamplingAdjustmentActivity, SystemPromptAssemblyActivity,
        ValidateActivity,
    };
    use weft_core::{
        ClassifierConfig, DomainsConfig, GatewayConfig, HookEvent, ModelEntry, ProviderConfig,
        RouterConfig, ServerConfig, WeftConfig, WireFormat,
    };
    use weft_providers::{Capability, Provider, ProviderRegistry};
    use weft_reactor::{
        ActivityRegistry, Reactor, ReactorConfig,
        config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, RetryPolicy},
        services::{ReactorHandle, Services},
    };
    use weft_router::test_support::StubRouter;

    // ── Minimal test infrastructure for server unit tests ──────────────────
    //
    // Integration tests and their full infrastructure live in tests/http_integration.rs.
    // These helpers provide just enough to exercise the request handler for
    // validation and error-mapping unit tests.

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
            event_log: None,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    fn make_unit_test_router(llm: impl Provider + 'static) -> Router {
        let config = test_config();
        let mut providers = HashMap::new();
        providers.insert(
            "test-model".to_string(),
            Arc::new(llm) as Arc<dyn weft_providers::Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("test-model".to_string(), "claude-test".to_string());
        let mut max_tokens_map = HashMap::new();
        max_tokens_map.insert("test-model".to_string(), 1024u32);
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "test-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        let provider_registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens_map,
            caps,
            "test-model".to_string(),
        ));

        // Unpack Arc<WeftConfig> to build ConfigStore (takes owned WeftConfig).
        // Clone first so config is still available for BudgetConfig/ReactorConfig below.
        let weft_config = (*config).clone();
        let config_store = Arc::new(weft_core::ConfigStore::new(weft_config));
        let resolved_config = config_store.snapshot();
        let services = Arc::new(Services {
            config_store,
            resolved_config,
            providers: provider_registry as Arc<dyn weft_providers::ProviderService + Send + Sync>,
            router: Arc::new(StubRouter) as Arc<dyn weft_router::SemanticRouter + Send + Sync>,
            commands: Arc::new(weft_commands::test_support::StubCommandRegistry::new())
                as Arc<dyn weft_commands::CommandRegistry + Send + Sync>,
            memory: None,
            hooks: Arc::new(weft_hooks::HookRegistry::empty())
                as Arc<dyn weft_hooks::HookRunner + Send + Sync>,
            reactor_handle: std::sync::OnceLock::new(),
            request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        });

        let mut registry = ActivityRegistry::new();
        registry.register(Arc::new(ValidateActivity)).unwrap();
        registry.register(Arc::new(ModelSelectionActivity)).unwrap();
        registry
            .register(Arc::new(CommandSelectionActivity))
            .unwrap();
        registry
            .register(Arc::new(ProviderResolutionActivity))
            .unwrap();
        registry
            .register(Arc::new(SystemPromptAssemblyActivity))
            .unwrap();
        registry
            .register(Arc::new(CommandFormattingActivity))
            .unwrap();
        registry
            .register(Arc::new(SamplingAdjustmentActivity))
            .unwrap();
        registry.register(Arc::new(GenerateActivity)).unwrap();
        registry.register(Arc::new(ExecuteCommandActivity)).unwrap();
        registry
            .register(Arc::new(AssembleResponseActivity))
            .unwrap();
        for event in [
            HookEvent::RequestStart,
            HookEvent::RequestEnd,
            HookEvent::PreResponse,
            HookEvent::PreToolUse,
            HookEvent::PostToolUse,
        ] {
            // Conservative aggregation: if ANY hook for this event has critical: true,
            // the entire HookActivity for that event is critical.
            let critical = config
                .hooks
                .iter()
                .filter(|h| h.event == event)
                .any(|h| h.critical);
            registry
                .register(Arc::new(HookActivity::new(
                    event,
                    Arc::clone(&services.hooks),
                    Arc::clone(&services.request_end_semaphore),
                    critical,
                )))
                .unwrap();
        }

        let event_log: Arc<dyn weft_reactor::event_log::EventLog> =
            Arc::new(weft_eventlog_memory::InMemoryEventLog::new());

        let reactor_config = ReactorConfig {
            pipelines: vec![PipelineConfig {
                name: "default".to_string(),
                pre_loop: vec![
                    ActivityRef::Name("validate".to_string()),
                    ActivityRef::Name("hook_request_start".to_string()),
                    ActivityRef::Name("model_selection".to_string()),
                    ActivityRef::Name("command_selection".to_string()),
                    ActivityRef::Name("provider_resolution".to_string()),
                    ActivityRef::Name("system_prompt_assembly".to_string()),
                    ActivityRef::Name("command_formatting".to_string()),
                    ActivityRef::Name("sampling_adjustment".to_string()),
                ],
                post_loop: vec![
                    ActivityRef::Name("assemble_response".to_string()),
                    ActivityRef::Name("hook_request_end".to_string()),
                ],
                generate: ActivityRef::WithConfig {
                    name: "generate".to_string(),
                    config: serde_json::Value::Null,
                    retry: Some(RetryPolicy {
                        max_retries: 0,
                        initial_backoff_ms: 0,
                        max_backoff_ms: 0,
                        backoff_multiplier: 1.0,
                    }),
                    timeout_secs: Some(config.gateway.request_timeout_secs),
                    heartbeat_interval_secs: Some(15),
                },
                execute_command: ActivityRef::Name("execute_command".to_string()),
                loop_hooks: LoopHooks::default(),
            }],
            budget: BudgetConfig {
                max_generation_calls: 20,
                max_iterations: config.gateway.max_command_iterations,
                max_depth: 5,
                timeout_secs: config.gateway.request_timeout_secs,
                generation_timeout_secs: config.gateway.request_timeout_secs,
                command_timeout_secs: 10,
            },
        };

        let reactor = Arc::new(
            Reactor::new(
                Arc::clone(&services),
                event_log,
                Arc::new(registry),
                &reactor_config,
            )
            .expect("test reactor must construct"),
        );

        let handle = Arc::new(ReactorHandle::new(Arc::clone(&reactor)));
        services
            .reactor_handle
            .set(handle)
            .expect("OnceLock must be unset");

        let weft_service = Arc::new(WeftService::new(reactor, config));
        // Tests do not need Prometheus — pass None for the handle.
        build_router(weft_service, None)
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

    // ── Validation unit tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_invalid_json_returns_4xx() {
        let router = make_unit_test_router(weft_providers::test_support::StubProvider::new(
            "irrelevant",
        ));
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
        let router = make_unit_test_router(weft_providers::test_support::StubProvider::new(
            "irrelevant",
        ));
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
        let router = make_unit_test_router(weft_providers::test_support::StubProvider::new(
            "irrelevant",
        ));
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
        let router = make_unit_test_router(weft_providers::test_support::StubProvider::new(
            "irrelevant",
        ));
        let body = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": true
        });
        let (status, resp) = post_json(router, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(resp["error"]["message"].is_string());
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
        use weft_core::{ContentPart, Source, WeftResponse, WeftTiming, WeftUsage};
        let resp = WeftResponse {
            id: "chatcmpl-test".to_string(),
            model: "gpt-4".to_string(),
            messages: vec![weft_core::WeftMessage {
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
            degradations: vec![],
        };
        let openai_resp = weft_to_openai(resp, "gpt-4".to_string());
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
            degradations: vec![],
        };
        let openai_resp = weft_to_openai(resp, "auto".to_string());
        // No assistant message → empty content, not a panic
        assert_eq!(openai_resp.choices[0].message.content, "");
    }
}
