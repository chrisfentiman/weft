//! HTTP integration tests for the `weft` gateway.
//!
//! These tests exercise the full `Router` -> `WeftService` -> `Reactor` -> Activities path
//! for the OpenAI-compatible HTTP API. They are integration tests (not unit tests) because
//! they send requests through the complete tower service stack including all pre-loop
//! activities and the real event-log path.
//!
//! Unit tests for request validation, translation, and error mapping remain in
//! `src/server.rs` where they test pure functions in isolation.

mod harness;

use axum::{body::Body, http::Request, http::StatusCode};
use serde_json::json;
use tower::ServiceExt;
use weft::server::build_router;
use weft_providers::ProviderError;
use weft_providers::test_support::SingleUseErrorProvider;

use harness::{TestProvider, make_router, make_weft_service, post_json};

// ── HTTP integration tests ─────────────────────────────────────────────────────

/// A valid request with a single user message returns HTTP 200 with a chat completion
/// response body. Verifies end-to-end routing, generation, and response assembly.
#[tokio::test]
async fn test_valid_request_returns_200() {
    let router = make_router(TestProvider::ok("Hello there!"));
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let (status, resp) = post_json(router, body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["object"], "chat.completion");
    assert_eq!(resp["choices"][0]["message"]["content"], "Hello there!");
    assert!(resp["id"].as_str().unwrap().starts_with("chatcmpl-"));
}

/// An LLM provider failure propagates as HTTP 500. Verifies that provider errors
/// are mapped to internal server errors on the HTTP path.
#[tokio::test]
async fn test_llm_error_returns_500() {
    let router = make_router(SingleUseErrorProvider::new(ProviderError::RequestFailed(
        "internal".to_string(),
    )));
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let (status, _) = post_json(router, body).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
}

/// A rate-limited provider returns HTTP 429 with a `Retry-After` header. Verifies that
/// `ProviderError::RateLimited` is mapped to 429 with the correct retry-after value.
#[tokio::test]
async fn test_rate_limited_returns_429_with_retry_after() {
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = make_router(SingleUseErrorProvider::new(ProviderError::RateLimited {
        retry_after_ms: 2000,
    }))
    .oneshot(req)
    .await
    .unwrap();

    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    // retry_after_ms=2000 → Retry-After: 2 seconds
    assert_eq!(
        resp.headers()
            .get("Retry-After")
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
}

/// A successful response includes usage statistics. Verifies that token usage
/// from the provider is correctly surfaced in the OpenAI-format response body.
#[tokio::test]
async fn test_response_contains_usage() {
    let router = make_router(TestProvider::ok("Done"));
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let (status, resp) = post_json(router, body).await;
    assert_eq!(status, StatusCode::OK);
    let usage = &resp["usage"];
    assert_eq!(usage["prompt_tokens"], 5);
    assert_eq!(usage["completion_tokens"], 3);
    assert_eq!(usage["total_tokens"], 8);
}

/// The health endpoint returns HTTP 200 with the expected JSON shape. Verifies
/// the `/health` endpoint is wired and returns the required fields.
#[tokio::test]
async fn test_health_endpoint_returns_200() {
    let weft_service = make_weft_service(TestProvider::ok("irrelevant"));
    let router = build_router(std::sync::Arc::clone(&weft_service), None);

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
    let health: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(health["status"], "ok");
    assert!(health["classifier_loaded"].is_boolean());
    assert!(health["tool_registry_connected"].is_boolean());
    // grpc_service field must be present
    assert!(health["grpc_service"].is_string());
    assert_eq!(health["grpc_service"], "serving");
}

/// The `model` field from the request is echoed verbatim in the response, regardless
/// of which internal model handled it. Prevents the model name from being rewritten.
#[tokio::test]
async fn test_response_model_preserved_from_request() {
    let router = make_router(TestProvider::ok("Hi"));
    let body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let (status, resp) = post_json(router, body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["model"], "test-model");
}

/// Error responses use the OpenAI error envelope format. Verifies that error responses
/// include the `error.message`, `error.type`, and `error.code` fields.
#[tokio::test]
async fn test_error_body_format() {
    let router = make_router(SingleUseErrorProvider::new(ProviderError::RequestFailed(
        "internal".to_string(),
    )));
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let (_, resp) = post_json(router, body).await;

    // Must have the OpenAI error envelope
    assert!(resp["error"]["message"].is_string());
    assert!(resp["error"]["type"].is_string());
    // code is present (may be null)
    assert!(resp["error"].get("code").is_some());
}

/// Both HTTP and gRPC entry points share the same `WeftService` instance.
/// Verifies that `build_router` wires `Arc<WeftService>` correctly and that
/// the HTTP path successfully calls `handle_weft_request` on it.
#[tokio::test]
async fn test_shared_weft_service_single_code_path() {
    let weft_service = make_weft_service(TestProvider::ok("shared path response"));
    let router = build_router(std::sync::Arc::clone(&weft_service), None);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "test single code path"}]
    });
    let (status, resp) = post_json(router, body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        resp["choices"][0]["message"]["content"],
        "shared path response"
    );
}

/// Requests with `content-type: application/grpc+proto` are dispatched to the tonic
/// handler, not the axum HTTP handler. Verifies the gRPC routing layer is wired correctly
/// by checking that tonic's `content-type: application/grpc` header appears in the response.
///
/// Real gRPC requires HTTP/2 framing. A tower `oneshot` sends HTTP/1.1, so tonic rejects
/// the request with a gRPC status error — but the observable is the response's
/// `content-type: application/grpc` header, which proves tonic handled it.
#[tokio::test]
async fn test_grpc_content_type_routes_to_tonic_handler() {
    let router = make_router(TestProvider::ok("irrelevant"));

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

// ── Prometheus /metrics endpoint tests ────────────────────────────────────────

/// When a `PrometheusHandle` is supplied, `GET /metrics` returns HTTP 200 with
/// `content-type: text/plain` and Prometheus text-format body.
#[tokio::test]
async fn test_metrics_endpoint_returns_200_when_enabled() {
    use metrics_exporter_prometheus::PrometheusBuilder;

    // Install a fresh recorder for this test. Each test binary starts a new process,
    // so there is no collision with the global recorder from other tests.
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("PrometheusBuilder must succeed in integration tests");

    let router = build_router(
        make_weft_service(TestProvider::ok("irrelevant")),
        Some(handle),
    );

    let req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "/metrics must return 200 when Prometheus is enabled"
    );

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/plain"),
        "/metrics content-type must be text/plain, got: {content_type:?}"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = std::str::from_utf8(&bytes).expect("/metrics body must be valid UTF-8");

    // A valid Prometheus text-format response starts with "# HELP" lines or is empty.
    // We only verify it is valid UTF-8 and does not contain an error indicator.
    assert!(
        !body.contains("error"),
        "/metrics body must not contain error text, got: {body}"
    );
}

/// When no `PrometheusHandle` is supplied, `GET /metrics` does NOT return
/// a Prometheus text-format response (content-type is not `text/plain; version=0.0.4`).
///
/// The underlying gRPC router may still respond to the path (it catches all unmatched
/// routes), but it must not serve Prometheus metrics.
#[tokio::test]
async fn test_metrics_endpoint_not_prometheus_when_disabled() {
    let router = build_router(make_weft_service(TestProvider::ok("irrelevant")), None);

    let req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();

    // The /metrics route is absent from the axum HTTP router. Either the gRPC router
    // intercepts the request (no Prometheus content-type) or axum returns 404.
    // Either way, the Prometheus content-type header must not be present.
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        !content_type.starts_with("text/plain; version=0.0.4"),
        "/metrics must not serve Prometheus format when disabled, got content-type: {content_type:?}"
    );
}

/// After a complete chat completion request flows through the pipeline,
/// `GET /metrics` must contain non-zero `weft_requests_total` and
/// `weft_request_duration_seconds` lines.
///
/// Verifies that `MetricsLayer` actually populates metrics from real reactor spans,
/// not just that the `/metrics` endpoint exists. A Prometheus handle with zero metrics
/// returns an empty body that would pass the 200+content-type checks but not this test.
#[test]
fn test_metrics_populated_after_request() {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use pretty_assertions::assert_eq;
    use tracing_subscriber::{Registry, layer::SubscriberExt};
    use weft::telemetry::metrics_layer::MetricsLayer;

    // Build a non-global Prometheus recorder so this test does not conflict with
    // test_metrics_endpoint_returns_200_when_enabled when both run in the same
    // process (cargo test). build_recorder() returns a PrometheusRecorder that
    // we install as a thread-local recorder via metrics::with_local_recorder.
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();

    // The MetricsLayer observes span lifecycle events and emits metrics via the
    // `metrics` crate macros. Wire it into a tracing_subscriber Registry.
    let metrics_layer = MetricsLayer::new();
    let subscriber = Registry::default().with(metrics_layer);

    // Route all metrics writes in this thread to our local recorder, and
    // install the subscriber as the default for this thread. Both are
    // thread-local, which is safe because we drive the async runtime on a
    // single current_thread executor — all .await suspension points resume
    // on the same thread, so the thread-locals are always visible.
    metrics::with_local_recorder(&recorder, || {
        tracing::subscriber::with_default(subscriber, || {
            // Build a single-threaded Tokio runtime. current_thread keeps all
            // futures on this OS thread, preserving thread-local recorder and
            // subscriber state across every .await boundary.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio current_thread runtime must build");

            rt.block_on(async {
                // Build the router with our local Prometheus handle so that
                // GET /metrics renders from the same recorder that received
                // the span-derived metric writes above.
                let router = build_router(
                    make_weft_service(TestProvider::ok("metrics test response")),
                    Some(handle.clone()),
                );

                // Send a valid chat completion request through the full pipeline.
                // The reactor creates a `request` span during execution; MetricsLayer
                // emits weft_requests_total and weft_request_duration_seconds on span close.
                let (status, _resp) = post_json(
                    router.clone(),
                    json!({
                        "model": "auto",
                        "messages": [{"role": "user", "content": "hello metrics"}]
                    }),
                )
                .await;
                assert_eq!(
                    status,
                    axum::http::StatusCode::OK,
                    "chat completion must return 200 before checking metrics"
                );

                // Scrape the metrics endpoint.
                let metrics_req = Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap();
                let metrics_resp = tower::ServiceExt::oneshot(router, metrics_req)
                    .await
                    .unwrap();

                assert_eq!(
                    metrics_resp.status(),
                    axum::http::StatusCode::OK,
                    "/metrics must return 200"
                );

                let bytes = axum::body::to_bytes(metrics_resp.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let body =
                    std::str::from_utf8(&bytes).expect("/metrics body must be valid UTF-8");

                // weft_requests_total must appear and carry a non-zero value.
                // The Prometheus text format for a counter looks like:
                //   weft_requests_total{...} 1
                assert!(
                    body.contains("weft_requests_total"),
                    "weft_requests_total must be present in /metrics after a request; body was:\n{body}"
                );

                // Find the first data line for weft_requests_total (not a # HELP / # TYPE line)
                // and verify the value is non-zero.
                let total_value = body
                    .lines()
                    .filter(|l| l.starts_with("weft_requests_total{"))
                    .filter_map(|l| l.rsplit_once(' ').and_then(|(_, v)| v.trim().parse::<f64>().ok()))
                    .sum::<f64>();
                assert!(
                    total_value > 0.0,
                    "weft_requests_total must be > 0 after one request, got {total_value}"
                );

                // weft_request_duration_seconds histogram must also be present.
                assert!(
                    body.contains("weft_request_duration_seconds"),
                    "weft_request_duration_seconds must be present in /metrics after a request; body was:\n{body}"
                );
            });
        });
    });
}

// ── Anthropic compat endpoint tests ───────────────────────────────────────────

/// Helper: POST JSON to `/v1/messages` and return (status, parsed body).
async fn post_anthropic(
    router: axum::Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = tower::ServiceExt::oneshot(router, req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// A valid Anthropic Messages API request returns HTTP 200 with an Anthropic-shaped
/// response body. Verifies end-to-end translation through the Anthropic compat path.
#[tokio::test]
async fn test_anthropic_valid_request_returns_200() {
    let router = make_router(TestProvider::ok("Hello from Anthropic compat!"));
    let body = json!({
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1024
    });
    let (status, resp) = post_anthropic(router, body).await;
    assert_eq!(status, StatusCode::OK);
    // Must use Anthropic response shape, not OpenAI choices
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["role"], "assistant");
    assert_eq!(resp["stop_reason"], "end_turn");
    assert!(
        resp["choices"].is_null(),
        "response must not use OpenAI choices shape"
    );
}

/// The Anthropic compat response uses `content` array of content blocks,
/// not the OpenAI `choices` array.
#[tokio::test]
async fn test_anthropic_response_uses_content_blocks() {
    let router = make_router(TestProvider::ok("Response text here"));
    let body = json!({
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 1024
    });
    let (status, resp) = post_anthropic(router, body).await;
    assert_eq!(status, StatusCode::OK);
    // content[0].type must be "text"
    assert_eq!(resp["content"][0]["type"], "text");
    // content[0].text must contain the response
    assert_eq!(resp["content"][0]["text"], "Response text here");
}

/// The Anthropic compat response includes usage statistics with Anthropic field names.
#[tokio::test]
async fn test_anthropic_response_contains_usage() {
    let router = make_router(TestProvider::ok("Done"));
    let body = json!({
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1024
    });
    let (status, resp) = post_anthropic(router, body).await;
    assert_eq!(status, StatusCode::OK);
    // Anthropic field names: input_tokens and output_tokens (not prompt_tokens/completion_tokens)
    assert!(resp["usage"]["input_tokens"].is_number());
    assert!(resp["usage"]["output_tokens"].is_number());
}

/// The `model` field is echoed from the request in the Anthropic response.
#[tokio::test]
async fn test_anthropic_response_model_echoed() {
    let router = make_router(TestProvider::ok("Hi"));
    let body = json!({
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 512
    });
    let (status, resp) = post_anthropic(router, body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["model"], "claude-3-haiku-20240307");
}

/// Anthropic error responses use the Anthropic error shape, not the OpenAI shape.
#[tokio::test]
async fn test_anthropic_error_uses_anthropic_shape() {
    let router = make_router(TestProvider::ok("irrelevant"));
    // Empty messages → 400 with Anthropic error shape
    let body = json!({
        "model": "claude-3-opus-20240229",
        "messages": [],
        "max_tokens": 1024
    });
    let (status, resp) = post_anthropic(router, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // Anthropic error shape: { type: "error", error: { type: "...", message: "..." } }
    assert_eq!(resp["type"], "error");
    assert!(resp["error"]["type"].is_string());
    assert!(resp["error"]["message"].is_string());
    // Must NOT use OpenAI error shape
    assert!(
        resp["error"]["code"].is_null()
            || !resp
                .get("error")
                .and_then(|e| e.get("code"))
                .is_some_and(|c| c.is_string())
    );
}

/// The Anthropic compat endpoint and OpenAI compat endpoint work independently
/// on the same `WeftService` instance.
#[tokio::test]
async fn test_both_endpoints_work_independently() {
    let svc = make_weft_service(TestProvider::ok("response text"));
    let router = build_router(std::sync::Arc::clone(&svc), None);

    // OpenAI endpoint still works
    let openai_body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello OpenAI path"}]
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&openai_body).unwrap()))
        .unwrap();
    let resp = tower::ServiceExt::oneshot(router.clone(), req)
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Anthropic endpoint also works on the same service
    let anthropic_body = json!({
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hello Anthropic path"}],
        "max_tokens": 1024
    });
    let (anthropic_status, anthropic_resp) = post_anthropic(router, anthropic_body).await;
    assert_eq!(anthropic_status, StatusCode::OK);
    assert_eq!(anthropic_resp["type"], "message");
}
