//! Command loop integration tests for the `weft` gateway.
//!
//! These tests exercise the full command execution loop end-to-end through the real
//! Reactor, real Activities, and a controllable `SequencedProvider`:
//!
//! `Router` → `WeftService` → `Reactor` → `GenerateActivity` → `ExecuteCommandActivity`
//!   → result injection → `GenerateActivity` (second call) → `AssembleResponseActivity`
//!
//! The event log is inspected after each request to verify:
//! - Commands are detected, executed, and their results injected back.
//! - Multiple commands in a single response are each executed.
//! - Iteration limits terminate the loop gracefully.
//! - Command descriptions appear in the formatted prompt injection.

mod harness;

use axum::http::StatusCode;
use serde_json::json;

use harness::{
    ScriptedCommandRegistry, SequencedProvider, SharedSequencedProvider,
    make_router_with_event_log, make_weft_service_with_config, make_weft_service_with_event_log,
    post_json, test_config_with_gateway,
};
use weft::server::build_router;

// ── Helper ─────────────────────────────────────────────────────────────────────

/// Count occurrences of an event type in the sorted event log.
fn count_event_type(events: &[weft_reactor::event::Event], event_type: &str) -> usize {
    events.iter().filter(|e| e.event_type == event_type).count()
}

/// Find the first event matching `event_type`, or `None` if absent.
fn find_event<'a>(
    events: &'a [weft_reactor::event::Event],
    event_type: &str,
) -> Option<&'a weft_reactor::event::Event> {
    events.iter().find(|e| e.event_type == event_type)
}

/// Find all events matching `event_type` in sequence order.
fn find_events<'a>(
    events: &'a [weft_reactor::event::Event],
    event_type: &str,
) -> Vec<&'a weft_reactor::event::Event> {
    let mut matched: Vec<&weft_reactor::event::Event> = events
        .iter()
        .filter(|e| e.event_type == event_type)
        .collect();
    matched.sort_by_key(|e| e.sequence);
    matched
}

/// Sort all events by sequence and return just the `event_type` strings.
fn sorted_event_types(events: &[weft_reactor::event::Event]) -> Vec<String> {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|e| e.sequence);
    sorted.into_iter().map(|e| e.event_type).collect()
}

/// Assert that `required` appears in `types` strictly after `after_pos`.
/// Returns the new `last_pos` (one past the found element).
fn assert_event_after(types: &[String], required: &str, after_pos: usize) -> usize {
    let pos = types[after_pos..]
        .iter()
        .position(|t| t == required)
        .unwrap_or_else(|| {
            panic!(
                "expected event type '{}' after position {}, but not found.\nAll event types: {:?}",
                required, after_pos, types
            )
        });
    after_pos + pos + 1
}

// ── test_commands_formatted_and_injected ──────────────────────────────────────

/// Verifies that `CommandFormattingActivity` formats registered commands and injects
/// their descriptions into the conversation via a `"message.injected"` event with
/// source `CommandFormatInjection`.
///
/// **Scenario:** One registered command ("web_search"). Provider returns a simple text
/// response (no command invocation). We are only testing that command descriptions
/// reach the LLM via prompt injection, not the command loop itself.
///
/// **Assertions:**
/// - Response is 200 OK.
/// - Event log contains `"commands.formatted"` with a non-`NoCommands` format.
/// - Event log contains `"message.injected"` with source `CommandFormatInjection`.
/// - The injected message content includes "web_search" and "Search the web".
///
/// **Regression guard:** Prevents `CommandFormattingActivity` from silently producing
/// `NoCommands` despite commands being registered, which would hide commands from the LLM.
#[tokio::test]
async fn test_commands_formatted_and_injected() {
    use harness::TestProvider;

    let commands = ScriptedCommandRegistry::new(vec![("web_search", "Search the web", "results")]);
    let (router, event_log) =
        make_router_with_event_log(TestProvider::ok("No commands needed"), commands);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Search for something"}]
    });
    let (status, resp) = post_json(router, body).await;

    assert_eq!(status, StatusCode::OK, "expected 200 OK, got: {resp}");

    let events = event_log.all_events();

    // commands.formatted must be present.
    let formatted_event = find_event(&events, "commands.formatted")
        .expect("expected commands.formatted event in log");

    // The format must not be NoCommands — we registered a command.
    let format_val = &formatted_event.payload["CommandsFormatted"]["format"];
    assert_ne!(
        format_val.as_str().unwrap_or(""),
        "NoCommands",
        "commands.formatted must not be NoCommands when commands are registered; got payload: {}",
        formatted_event.payload
    );

    // message.injected with source CommandFormatInjection must be present.
    let injected_events = find_events(&events, "message.injected");
    let format_injection = injected_events
        .iter()
        .find(|e| e.payload["MessageInjected"]["source"] == "CommandFormatInjection")
        .unwrap_or_else(|| {
            panic!(
                "expected message.injected event with source CommandFormatInjection.\nAll message.injected payloads: {:?}",
                injected_events.iter().map(|e| &e.payload).collect::<Vec<_>>()
            )
        });

    // The injected content must include the command name and description.
    let content_str = format_injection.payload["MessageInjected"]["message"]["content"].to_string();
    assert!(
        content_str.contains("web_search"),
        "injected command text must contain 'web_search', got: {content_str}"
    );
    assert!(
        content_str.contains("Search the web"),
        "injected command text must contain 'Search the web', got: {content_str}"
    );
}

// ── test_command_loop_full_cycle ───────────────────────────────────────────────

/// Verifies the full command loop round-trip: LLM returns a command call, the
/// command executes, the result is injected back, and the LLM generates a final
/// text response.
///
/// **Scenario:**
/// - Call 1: Provider returns `WithCommands { text: "I'll search for that.", commands: [web_search] }`
/// - Command: `web_search` executes and returns "Results: Rust async guide".
/// - Call 2: Provider returns `Text("Based on the search results, here's what I found.")`
///
/// **Assertions:**
/// - Response is 200 OK with content containing "Based on the search results".
/// - Provider `call_count()` is exactly 2.
/// - Event log (in order): generation.started → generation.completed → command.started
///   → command.completed (success: true) → generation.started → generation.completed.
///
/// **Regression guard:** The most important test in this spec. Verifies the complete
/// LLM → command → LLM round-trip that was previously untested.
#[tokio::test]
async fn test_command_loop_full_cycle() {
    let commands = ScriptedCommandRegistry::new(vec![(
        "web_search",
        "Search the web",
        "Results: Rust async guide",
    )]);

    let provider = SharedSequencedProvider::new(SequencedProvider::new(vec![
        harness::MockResponse::WithCommands {
            text: "I'll search for that.".into(),
            commands: vec![("web_search".into(), json!({"query": "Rust async patterns"}))],
        },
        harness::MockResponse::Text("Based on the search results, here's what I found.".into()),
    ]));
    let provider_clone = provider.clone();

    let (svc, event_log) = make_weft_service_with_event_log(provider, commands);
    let router = build_router(svc);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell me about Rust async"}]
    });
    let (status, resp) = post_json(router, body).await;

    assert_eq!(status, StatusCode::OK, "expected 200 OK, got: {resp}");

    // Response text must come from the second provider call.
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        content.contains("Based on the search results"),
        "response must contain text from second provider call, got: {content}"
    );

    // Provider was called exactly twice: once for the command invocation, once for
    // the final response.
    let call_count = provider_clone.call_count();
    assert_eq!(
        call_count, 2,
        "expected exactly 2 provider calls, got {call_count}"
    );

    // Verify the event ordering in the log.
    let events = event_log.all_events();
    let types = sorted_event_types(&events);

    let mut pos = 0;
    pos = assert_event_after(&types, "generation.started", pos);
    pos = assert_event_after(&types, "generation.completed", pos);
    pos = assert_event_after(&types, "command.started", pos);
    pos = assert_event_after(&types, "command.completed", pos);
    pos = assert_event_after(&types, "generation.started", pos);
    assert_event_after(&types, "generation.completed", pos);

    // command.completed must carry success: true.
    let cmd_completed =
        find_event(&events, "command.completed").expect("command.completed event must exist");
    let success = cmd_completed.payload["CommandCompleted"]["result"]["success"]
        .as_bool()
        .unwrap_or(false);
    assert!(
        success,
        "command.completed must carry success: true, got payload: {}",
        cmd_completed.payload
    );

    // command.started must name the correct command.
    let cmd_started =
        find_event(&events, "command.started").expect("command.started event must exist");
    let cmd_name = cmd_started.payload["CommandStarted"]["invocation"]["name"]
        .as_str()
        .unwrap_or("");
    assert_eq!(
        cmd_name, "web_search",
        "command.started must name web_search, got: {cmd_name}"
    );
}

// ── test_command_loop_respects_iteration_limit ────────────────────────────────

/// Verifies that the Reactor enforces `max_command_iterations` and terminates
/// the command loop gracefully when the LLM keeps returning command calls.
///
/// **Scenario:** `max_command_iterations = 2`. Provider always returns a command call
/// (never a plain text response). The Reactor must stop the loop after 2 command
/// iterations and return a response (not hang or 500).
///
/// **Assertions:**
/// - Response is 200 OK (budget exhaustion is graceful, not an error).
/// - Provider `call_count()` is at most 3 (initial call + up to 2 command iterations).
/// - Event log contains `"budget.exhausted"` indicating the limit was hit.
/// - Event log contains at most 2 `"iteration.completed"` events.
///
/// **Regression guard:** Without iteration limits, a command-invoking LLM could loop
/// indefinitely. This test verifies the real Reactor enforces the budget.
#[tokio::test]
async fn test_command_loop_respects_iteration_limit() {
    let config = test_config_with_gateway("You are helpful.", 2);
    let commands = ScriptedCommandRegistry::new(vec![("search", "Search", "result")]);

    // Every response is a command call — the loop must terminate via budget, not text response.
    let provider = SharedSequencedProvider::new(SequencedProvider::new(vec![
        harness::MockResponse::WithCommands {
            text: String::new(),
            commands: vec![("search".into(), json!({}))],
        },
        harness::MockResponse::WithCommands {
            text: String::new(),
            commands: vec![("search".into(), json!({}))],
        },
        harness::MockResponse::WithCommands {
            text: String::new(),
            commands: vec![("search".into(), json!({}))],
        },
    ]));
    let provider_clone = provider.clone();

    let (svc, event_log) = make_weft_service_with_config(config, provider, commands);
    let router = build_router(svc);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Search repeatedly"}]
    });
    let (status, _resp) = post_json(router, body).await;

    // Budget exhaustion must be graceful — the Reactor assembles whatever partial
    // response it has rather than returning a 500.
    assert_eq!(
        status,
        StatusCode::OK,
        "budget exhaustion must return 200 OK, not an error"
    );

    let call_count = provider_clone.call_count();
    assert!(
        call_count <= 3,
        "provider must be called at most 3 times (initial + 2 iterations), got {call_count}"
    );

    let events = event_log.all_events();

    // budget.exhausted must appear — this is the Reactor's signal that the limit was hit.
    let has_budget_exhausted = events.iter().any(|e| e.event_type == "budget.exhausted");
    assert!(
        has_budget_exhausted,
        "expected budget.exhausted event when iteration limit is hit.\nAll event types: {:?}",
        sorted_event_types(&events)
    );

    // At most 2 iteration.completed events (one per completed command iteration).
    let iteration_count = count_event_type(&events, "iteration.completed");
    assert!(
        iteration_count <= 2,
        "expected at most 2 iteration.completed events, got {iteration_count}"
    );
}

// ── test_multiple_commands_in_single_response ─────────────────────────────────

/// Verifies that the Reactor executes all commands when the LLM returns multiple
/// command calls in a single response.
///
/// **Scenario:**
/// - Call 1: Provider returns `WithCommands` with two commands: `web_search` and `calculator`.
/// - Both commands execute with their configured outputs.
/// - Call 2: Provider returns a final text response referencing both results.
///
/// **Assertions:**
/// - Response is 200 OK.
/// - Provider `call_count()` is exactly 2 (one call per loop iteration).
/// - Two `"command.completed"` events in the log, one for each command.
/// - Both command names appear in the `command.started` events.
///
/// **Regression guard:** LLMs frequently invoke multiple commands in a single response.
/// If the Reactor only executes the first command, users get incomplete results.
#[tokio::test]
async fn test_multiple_commands_in_single_response() {
    let commands = ScriptedCommandRegistry::new(vec![
        ("web_search", "Search the web", "Search results here"),
        ("calculator", "Evaluate math", "42"),
    ]);

    let provider = SharedSequencedProvider::new(SequencedProvider::new(vec![
        harness::MockResponse::WithCommands {
            text: "Let me search and calculate.".into(),
            commands: vec![
                ("web_search".into(), json!({})),
                ("calculator".into(), json!({"expr": "6*7"})),
            ],
        },
        harness::MockResponse::Text("The search found X and 6*7 is 42.".into()),
    ]));
    let provider_clone = provider.clone();

    let (svc, event_log) = make_weft_service_with_event_log(provider, commands);
    let router = build_router(svc);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Search and calculate"}]
    });
    let (status, resp) = post_json(router, body).await;

    assert_eq!(status, StatusCode::OK, "expected 200 OK, got: {resp}");

    // Response must come from the second call.
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        content.contains("42"),
        "final response must contain text from second provider call, got: {content}"
    );

    // Provider was called exactly twice: one for the multi-command response, one for
    // the final text response.
    let call_count = provider_clone.call_count();
    assert_eq!(
        call_count, 2,
        "expected exactly 2 provider calls, got {call_count}"
    );

    let events = event_log.all_events();

    // Both commands must have been executed — two command.completed events.
    let completed_count = count_event_type(&events, "command.completed");
    assert_eq!(
        completed_count, 2,
        "expected 2 command.completed events (one per command), got {completed_count}"
    );

    // Find the names of executed commands from command.started events.
    let started_events = find_events(&events, "command.started");
    assert_eq!(
        started_events.len(),
        2,
        "expected 2 command.started events, got {}",
        started_events.len()
    );

    let executed_names: Vec<&str> = started_events
        .iter()
        .filter_map(|e| e.payload["CommandStarted"]["invocation"]["name"].as_str())
        .collect();

    assert!(
        executed_names.contains(&"web_search"),
        "web_search must be in executed commands, got: {:?}",
        executed_names
    );
    assert!(
        executed_names.contains(&"calculator"),
        "calculator must be in executed commands, got: {:?}",
        executed_names
    );
}
