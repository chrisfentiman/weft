//! Error path and edge case integration tests for the `weft` gateway.
//!
//! These tests exercise the full pipeline stack through failure scenarios:
//! - Provider errors (LLM failure → HTTP 500)
//! - Command execution failures (logical failure → injected as context, loop continues)
//! - Edge cases (empty provider response → HTTP 200)
//! - Sampling parameter clamping (max_tokens > model limit → clamped, temperature passes through)
//!
//! The event log is inspected to verify that the Reactor and Activities handle
//! errors correctly — either propagating them as HTTP errors or feeding them
//! back to the LLM as context.

mod harness;

use axum::http::StatusCode;
use serde_json::json;

use harness::{
    ScriptedCommandRegistry, SequencedProvider, SharedSequencedProvider, TestProvider,
    make_router_with_event_log, make_weft_service_with_event_log, post_json,
};
use weft::server::build_router;
use weft_llm::{ProviderError, test_support::SingleUseErrorProvider};

// ── Helper ─────────────────────────────────────────────────────────────────────

/// Sort events by sequence and return the `event_type` strings.
fn sorted_event_types(events: &[weft_reactor::event::Event]) -> Vec<String> {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|e| e.sequence);
    sorted.into_iter().map(|e| e.event_type).collect()
}

/// Count occurrences of an event type.
fn count_event_type(events: &[weft_reactor::event::Event], event_type: &str) -> usize {
    events.iter().filter(|e| e.event_type == event_type).count()
}

// ── test_provider_failure_returns_500 ─────────────────────────────────────────

/// Verifies that when the LLM provider returns an error, the gateway returns HTTP 500
/// and the event log records a `"generation.failed"` event.
///
/// **Scenario:** `SingleUseErrorProvider` returns `ProviderError::RequestFailed` on the
/// first call. No command loop. The pipeline must propagate this failure as an HTTP 500.
///
/// **Assertions:**
/// - HTTP response is 500.
/// - Event log contains `"generation.failed"`.
///
/// **Regression guard:** Prevents provider errors from being silently swallowed or
/// incorrectly mapped to 200 OK responses with empty content.
#[tokio::test]
async fn test_provider_failure_returns_500() {
    let provider =
        SingleUseErrorProvider::new(ProviderError::RequestFailed("provider down".into()));
    let commands = weft_commands::test_support::StubCommandRegistry::new();
    let (router, event_log) = make_router_with_event_log(provider, commands);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let (status, _resp) = post_json(router, body).await;

    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "expected 500 when provider fails"
    );

    // generation.failed must be in the event log.
    let events = event_log.all_events();
    assert!(
        events.iter().any(|e| e.event_type == "generation.failed"),
        "expected generation.failed event in log after provider error.\nAll event types: {:?}",
        sorted_event_types(&events)
    );
}

// ── test_command_failure_injected_and_pipeline_continues ─────────────────────

/// Verifies that a command logical failure (`success: false`) is fed back to the LLM
/// as context, not treated as a pipeline error. The loop continues and the LLM
/// generates a final response.
///
/// **Scenario:**
/// - Call 1: Provider invokes "search" command.
/// - Command: "search" returns `CommandResult { success: false, error: "service unavailable" }`.
/// - Call 2: Provider returns a final text response acknowledging the failure.
///
/// **Assertions:**
/// - Response is 200 OK (command failure is NOT a pipeline failure).
/// - Event log contains `"command.completed"` with `success: false` for "search".
/// - A second `"generation.started"` event exists (pipeline continued after failure).
/// - Provider `call_count()` is 2 (did not abort).
///
/// **Regression guard:** Command failures must be fed back to the LLM as context so
/// it can acknowledge them or try alternatives. Treating them as pipeline errors
/// would hide transient service failures from the user.
#[tokio::test]
async fn test_command_failure_injected_and_pipeline_continues() {
    let commands = ScriptedCommandRegistry::with_failing_command(
        vec![("search", "Search", "")],
        "search",
        "service unavailable",
    );

    let provider = SharedSequencedProvider::new(SequencedProvider::new(vec![
        harness::MockResponse::WithCommands {
            text: String::new(),
            commands: vec![("search".into(), json!({}))],
        },
        harness::MockResponse::Text("The search failed, but I can still help.".into()),
    ]));
    let provider_clone = provider.clone();

    let (svc, event_log) = make_weft_service_with_event_log(provider, commands);
    let router = build_router(svc, None);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Search for something"}]
    });
    let (status, resp) = post_json(router, body).await;

    // Command failure must not abort the pipeline.
    assert_eq!(
        status,
        StatusCode::OK,
        "command failure must not cause a 500, got: {resp}"
    );

    let events = event_log.all_events();
    let types = sorted_event_types(&events);

    // command.completed must be present — the command was executed, it just failed logically.
    let cmd_completed = events
        .iter()
        .find(|e| e.event_type == "command.completed")
        .unwrap_or_else(|| {
            panic!(
                "expected command.completed event.\nAll event types: {:?}",
                types
            )
        });

    // The result must carry success: false.
    let success = cmd_completed.payload["event"]["result"]["success"]
        .as_bool()
        .unwrap_or(true);
    assert!(
        !success,
        "command.completed must carry success: false for a failing command, got payload: {}",
        cmd_completed.payload
    );

    // A second generation.started event must exist — the pipeline called the LLM again
    // after injecting the command failure as context.
    let gen_started_count = count_event_type(&events, "generation.started");
    assert!(
        gen_started_count >= 2,
        "expected at least 2 generation.started events (pipeline continued after command failure), got {gen_started_count}"
    );

    // Provider was called twice.
    let call_count = provider_clone.call_count();
    assert_eq!(
        call_count, 2,
        "expected exactly 2 provider calls, got {call_count}"
    );
}

// ── test_empty_provider_response_returns_200 ─────────────────────────────────

/// Verifies that an empty LLM response (empty string content) does not cause a
/// pipeline error. The gateway returns 200 OK with empty or minimal content.
///
/// **Scenario:** `TestProvider::ok("")` returns a response with empty text. The
/// pipeline must assemble and return the response without error.
///
/// **Assertions:**
/// - HTTP response is 200 OK (not a 500).
/// - Event log contains `"execution.completed"` (not `"execution.failed"`).
///
/// **Regression guard:** Prevents defensive code from treating empty responses as
/// errors and unnecessarily returning 500 to callers.
#[tokio::test]
async fn test_empty_provider_response_returns_200() {
    let commands = weft_commands::test_support::StubCommandRegistry::new();
    let (router, event_log) = make_router_with_event_log(TestProvider::ok(""), commands);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let (status, _resp) = post_json(router, body).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "empty provider response must return 200 OK, not an error"
    );

    let events = event_log.all_events();

    assert!(
        events.iter().any(|e| e.event_type == "execution.completed"),
        "expected execution.completed event for empty provider response.\nAll event types: {:?}",
        sorted_event_types(&events)
    );
    assert!(
        !events.iter().any(|e| e.event_type == "execution.failed"),
        "unexpected execution.failed event for empty provider response"
    );
}

// ── test_sampling_parameters_clamped_and_passed ───────────────────────────────

/// Verifies that `SamplingAdjustmentActivity` clamps `max_tokens` to the model limit
/// and passes `temperature` through unchanged.
///
/// **Scenario:** Request specifies `temperature: 0.7` and `max_tokens: 2048`. The
/// test model has `max_tokens: 1024` (set by `test_config`). The activity must
/// clamp `max_tokens` to 1024 before passing the request to the provider.
///
/// **Assertions:**
/// - Response is 200 OK.
/// - Event log `"sampling.updated"` event has `max_tokens <= 1024`.
/// - `provider.last_request()` has `options.temperature == Some(0.7)`.
/// - `provider.last_request()` has `options.max_tokens <= Some(1024)`.
///
/// **Regression guard:** Prevents sampling parameters from exceeding model limits,
/// which would cause provider API errors at runtime.
#[tokio::test]
async fn test_sampling_parameters_clamped_and_passed() {
    use weft_llm::ProviderRequest;

    let commands = weft_commands::test_support::StubCommandRegistry::new();
    let provider = SharedSequencedProvider::new(SequencedProvider::single("Done"));
    let provider_clone = provider.clone();

    let (svc, event_log) = make_weft_service_with_event_log(provider, commands);
    let router = build_router(svc, None);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "max_tokens": 2048
    });
    let (status, resp) = post_json(router, body).await;

    assert_eq!(status, StatusCode::OK, "expected 200 OK, got: {resp}");

    // Event log must contain context.sampling_updated with clamped max_tokens.
    let events = event_log.all_events();
    let sampling_event = events
        .iter()
        .find(|e| e.event_type == "context.sampling_updated")
        .unwrap_or_else(|| {
            panic!(
                "expected context.sampling_updated event.\nAll event types: {:?}",
                sorted_event_types(&events)
            )
        });

    let max_tokens_in_event = sampling_event.payload["event"]["max_tokens"]
        .as_u64()
        .expect("context.sampling_updated payload must have event.max_tokens");

    assert!(
        max_tokens_in_event <= 1024,
        "sampling.updated must clamp max_tokens to model limit 1024, got {max_tokens_in_event}"
    );

    // Provider last_request must carry the temperature through unchanged.
    let last_req = provider_clone
        .last_request()
        .expect("provider must have received at least one request");

    let ProviderRequest::ChatCompletion { options, .. } = last_req;

    // Temperature must pass through unchanged.
    let temp = options
        .temperature
        .expect("temperature must be set in provider request");
    assert!(
        (temp - 0.7_f32).abs() < 1e-4,
        "temperature must pass through to provider unchanged (0.7), got {temp}"
    );

    // max_tokens in the provider request must also be clamped.
    if let Some(mt) = options.max_tokens {
        assert!(
            mt <= 1024,
            "max_tokens in provider request must be clamped to 1024, got {mt}"
        );
    }
    // If max_tokens is None in the provider request, the provider uses its default
    // (which is within its own limit), so this is also acceptable.
}
