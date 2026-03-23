//! Pipeline integration tests for the `weft` gateway.
//!
//! These tests exercise the full pre-loop activity pipeline through HTTP:
//! `Router` → `WeftService` → `Reactor` → real Activities → `InMemoryEventLog`.
//!
//! Unlike the reactor dispatch tests (which use stub activities) and the HTTP integration
//! tests (which verify HTTP-layer behaviour), these tests verify that the real activity
//! implementations produce the expected event traces and that they compose correctly.
//!
//! The event log is inspected after each request to verify:
//! - The correct event types appear in order.
//! - Event payloads carry the expected field values.
//!
//! All test scenarios use a single-turn provider (no command loop) so the assertions
//! focus on the pre-loop activities only.

mod harness;

use axum::http::StatusCode;
use serde_json::json;

use harness::{
    ScriptedCommandRegistry, TestProvider, make_router_with_event_log,
    make_weft_service_with_event_log, post_json, test_config,
};
use weft::server::build_router;

// ── Helper ─────────────────────────────────────────────────────────────────────

/// Extract all event-type strings from the log in ascending sequence order.
///
/// `all_events()` does not guarantee order across executions but for a single
/// request there is exactly one execution in the log. Sorting by sequence gives
/// the canonical order for assertions.
fn sorted_event_types(events: &[weft_reactor::event::Event]) -> Vec<String> {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|e| e.sequence);
    sorted.into_iter().map(|e| e.event_type).collect()
}

// ── test_pre_loop_activities_produce_events_in_order ──────────────────────────

/// Verifies that all six pre-loop activities (validate, model_selection,
/// command_selection, provider_resolution, system_prompt_assembly,
/// command_formatting, sampling_adjustment) produce their expected events in
/// the correct order, followed by the generate/assemble/complete sequence.
///
/// Uses a single-response `TestProvider` so the pipeline does not enter the
/// command loop. The event log is the observable: if any activity fails to emit
/// its event or emits it in the wrong order, this test fails.
///
/// **Regression guard:** Prevents silent loss of activity events when the Reactor
/// or activity implementations change.
#[tokio::test]
async fn test_pre_loop_activities_produce_events_in_order() {
    let commands = ScriptedCommandRegistry::new(vec![("web_search", "Search the web", "results")]);
    let (router, event_log) = make_router_with_event_log(TestProvider::ok("Hello!"), commands);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let (status, resp) = post_json(router, body).await;

    // Request must succeed.
    assert_eq!(status, StatusCode::OK, "expected 200 OK, got: {resp}");
    assert_eq!(
        resp["choices"][0]["message"]["content"], "Hello!",
        "response text must come from TestProvider"
    );

    // Inspect the event trace.
    let events = event_log.all_events();
    assert!(
        !events.is_empty(),
        "event log must not be empty after a request"
    );

    let types = sorted_event_types(&events);

    // Assert that the key event types appear in the correct relative order.
    // There will be additional events in between (activity.completed, commands.available, etc.)
    // so we check position-based ordering, not exact sequence equality.
    let required_in_order: &[&str] = &[
        "execution.started",
        "activity.started", // validate
        "execution.validation_passed",
        "selection.model_selected",
        "selection.commands_selected",
        "selection.provider_resolved",
        "context.system_prompt_assembled",
        "context.commands_formatted",
        "context.sampling_updated",
        "generation.started",
        "generation.chunk", // at least one (Content or Done)
        "generation.completed",
        "context.response_assembled",
        "execution.completed",
    ];

    let mut last_pos = 0usize;
    for &required_type in required_in_order {
        let pos = types[last_pos..].iter().position(|t| t == required_type);
        assert!(
            pos.is_some(),
            "expected event type '{}' after position {}, but it was not found in event log.\nAll event types in order: {:?}",
            required_type,
            last_pos,
            types
        );
        last_pos += pos.unwrap() + 1;
    }

    // The provider was called exactly once: verify via a single "generation.started" event.
    let generation_started_count = types
        .iter()
        .filter(|t| t.as_str() == "generation.started")
        .count();
    assert_eq!(
        generation_started_count, 1,
        "expected exactly 1 generation.started event (provider called once), got {generation_started_count}"
    );

    // The execution must have completed (not failed or cancelled).
    assert!(
        types.contains(&"execution.completed".to_string()),
        "expected execution.completed event"
    );
    assert!(
        !types.contains(&"execution.failed".to_string()),
        "unexpected execution.failed event"
    );
}

// ── test_system_prompt_layers_gateway_and_caller ──────────────────────────────

/// Verifies that `SystemPromptAssemblyActivity` fires, assembles the gateway system prompt,
/// and injects it into the conversation as a `MessageInjected` event.
///
/// **Context on layering via HTTP:**
/// The OpenAI-compat HTTP translation (`weft_providers::openai::translate::parse_inbound_request`) assigns `Source::Gateway` to
/// incoming system messages. `SystemPromptAssemblyActivity` treats messages with
/// `Source::Client` as a second "caller layer". Since HTTP system messages arrive as
/// `Source::Gateway`, they are not counted as a second layer — only the gateway's own
/// config prompt is assembled. This is expected behaviour for the HTTP path.
///
/// The test verifies:
/// - Response is 200 OK.
/// - `system_prompt.assembled` event is present and carries `message_count >= 1`.
/// - The `message.injected` event with `source: SystemPromptAssembly` carries the gateway
///   prompt text ("You are a test assistant.") in its content.
///
/// **Regression guard:** Prevents `SystemPromptAssemblyActivity` from silently producing an
/// empty prompt or emitting no event.
#[tokio::test]
async fn test_system_prompt_layers_gateway_and_caller() {
    // test_config() sets gateway.system_prompt = "You are a test assistant."
    let config = test_config();
    let gateway_prompt = config.gateway.system_prompt.clone();

    let commands = weft_commands::test_support::StubCommandRegistry::new();
    let (svc, event_log) = make_weft_service_with_event_log(TestProvider::ok("Response"), commands);
    let router = build_router(svc, None);

    // Request with caller-supplied system message.
    // Via HTTP, this gets Source::Gateway (see parse_inbound_request), so it is NOT treated as
    // a second caller layer by SystemPromptAssemblyActivity. Only the gateway config prompt
    // is assembled; message_count reflects the post-injection state.
    let body = json!({
        "model": "auto",
        "messages": [
            {"role": "system", "content": "Always be concise."},
            {"role": "user", "content": "Hello"}
        ]
    });
    let (status, resp) = post_json(router, body).await;

    assert_eq!(status, StatusCode::OK, "expected 200 OK, got: {resp}");

    // Inspect the event log.
    let events = event_log.all_events();
    let mut sorted = events.clone();
    sorted.sort_by_key(|e| e.sequence);

    // context.system_prompt_assembled must be present.
    // PipelineEvent is adjacently tagged: outer has {"category": "Context", "event": {...}},
    // inner event has {"type": "SystemPromptAssembled", "message_count": N}.
    // Note: prompt_length and layer_count are observability-only and removed from the variant
    // per Phase 2 slimming; they are no longer stored in the event payload.
    let assembled_event = sorted
        .iter()
        .find(|e| e.event_type == "context.system_prompt_assembled")
        .expect("expected context.system_prompt_assembled event in log");

    // message_count must be >= 1 (at minimum the injected system message is counted).
    let message_count = assembled_event.payload["event"]["message_count"]
        .as_u64()
        .expect("context.system_prompt_assembled payload must have event.message_count field");
    assert!(
        message_count >= 1,
        "expected message_count >= 1, got {message_count}"
    );

    // Find the context.message_injected event with source = SystemPromptAssembly.
    // Payload shape: {"category": "Context", "event": {"type": "MessageInjected", "message": {...}, "source": "SystemPromptAssembly"}}
    let injected_event = sorted
        .iter()
        .find(|e| {
            e.event_type == "context.message_injected"
                && e.payload["event"]["source"] == "SystemPromptAssembly"
        })
        .expect("expected context.message_injected event with source SystemPromptAssembly in log");

    // The assembled message content must include the gateway prompt.
    let content_json = &injected_event.payload["event"]["message"]["content"];
    let content_str = content_json.to_string();

    assert!(
        content_str.contains(&gateway_prompt),
        "assembled system prompt must contain gateway prompt '{gateway_prompt}', but got: {content_str}"
    );
}
