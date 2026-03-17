/// Tests for handle_request and run_loop.
use super::*;
use std::time::Duration;
use weft_core::{ContentPart, GatewayConfig, Role, Source, WeftError, WeftMessage, WeftRequest};
use weft_llm::{Provider, ProviderError, ProviderRequest, ProviderResponse};

// ── Test: no-command response (single pass) ────────────────────────────

#[tokio::test]
async fn test_no_command_response_single_pass() {
    let registry = single_model_registry(
        MockLlmProvider::single("Hello, I can help you with that!"),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Hello, I can help you with that!");
    assert!(resp.id.starts_with("chatcmpl-"));
}

// ── Test: single command loop ──────────────────────────────────────────

#[tokio::test]
async fn test_single_command_executes_and_loops() {
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search query: \"Rust async\"",
            "Here are the results I found.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "Found 5 results about Rust async".to_string(),
                error: None,
            },
        ),
    );

    let resp = engine
        .handle_request(make_user_request("Search for Rust async"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Here are the results I found.");
}

// ── Test: multiple commands ────────────────────────────────────────────

#[tokio::test]
async fn test_multiple_commands_executed_sequentially() {
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search query: \"Rust\"\n/code_review target: src",
            "Done with both commands.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![
            ("web_search", "Search the web"),
            ("code_review", "Review code"),
        ])
        .with_execute_result(CommandResult {
            command_name: "any".to_string(),
            success: true,
            output: "OK".to_string(),
            error: None,
        }),
    );

    let resp = engine
        .handle_request(make_user_request("Do stuff"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Done with both commands.");
}

// ── Test: max iterations exceeded ─────────────────────────────────────

#[tokio::test]
async fn test_max_iterations_exceeded() {
    let config = Arc::new(WeftConfig {
        gateway: GatewayConfig {
            system_prompt: "test".to_string(),
            max_command_iterations: 2,
            request_timeout_secs: 30,
        },
        ..(*test_config()).clone()
    });

    let registry = single_model_registry(
        MockLlmProvider::single("/web_search query: \"loop\""),
        "test-model",
        "claude-test",
    );

    let engine = make_engine_with_config(
        config,
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "still looping".to_string(),
                error: None,
            },
        ),
    );

    let result = engine
        .handle_request(make_user_request("Search forever"))
        .await;

    assert!(
        matches!(result, Err(WeftError::CommandLoopExceeded { max: 2 })),
        "expected CommandLoopExceeded, got: {:?}",
        result
    );
}

// ── Test: command failure recovery ────────────────────────────────────

#[tokio::test]
async fn test_command_failure_injected_as_error_result() {
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search query: \"Rust\"",
            "The search failed, but I can still answer.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: false,
                output: String::new(),
                error: Some("registry unavailable".to_string()),
            },
        ),
    );

    let resp = engine
        .handle_request(make_user_request("Search for Rust"))
        .await
        .expect("should succeed despite command failure");

    assert_eq!(
        resp_text(&resp),
        "The search failed, but I can still answer."
    );
}

// ── Test: router fallback ──────────────────────────────────────────────

#[tokio::test]
async fn test_router_failure_falls_back_to_all_commands() {
    let registry = single_model_registry(
        MockLlmProvider::single("No commands needed."),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::failing(),
        MockCommandRegistry::new(vec![
            ("web_search", "Search the web"),
            ("code_review", "Review code"),
        ]),
    );

    // Should succeed even with a failing router (fallback to all commands)
    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("router fallback must not fail the request");

    assert!(!resp_text(&resp).is_empty());
}

// ── Test: --describe flag ──────────────────────────────────────────────

#[tokio::test]
async fn test_describe_action_handled_by_registry() {
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search --describe",
            "Now I understand the command.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_describe_result(
            CommandDescription {
                name: "web_search".to_string(),
                description: "Search the web for current information".to_string(),
                usage: "/web_search query: \"search terms\"".to_string(),
                parameters_schema: None,
            },
        ),
    );

    let resp = engine
        .handle_request(make_user_request("What is web_search?"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Now I understand the command.");
}

// ── Test: request timeout ──────────────────────────────────────────────

#[tokio::test]
async fn test_request_timeout() {
    let config = Arc::new(WeftConfig {
        gateway: GatewayConfig {
            system_prompt: "test".to_string(),
            max_command_iterations: 10,
            request_timeout_secs: 0,
        },
        ..(*test_config()).clone()
    });

    /// A provider that sleeps forever before responding.
    struct SlowLlmProvider;

    #[async_trait]
    impl Provider for SlowLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text("never".to_string())],
                    delta: false,
                    message_index: 0,
                },
                usage: None,
            })
        }

        fn name(&self) -> &str {
            "slow"
        }
    }

    let registry = single_model_registry(SlowLlmProvider, "test-model", "claude-test");
    let engine = make_engine_with_config(
        config,
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(result, Err(WeftError::RequestTimeout { .. })),
        "expected RequestTimeout, got: {:?}",
        result
    );
}

// ── Test: response format matches OpenAI schema ────────────────────────

#[tokio::test]
async fn test_response_format_matches_openai_schema() {
    let registry = single_model_registry(
        MockLlmProvider::single("Test response"),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let resp = engine
        .handle_request(make_user_request("Hi"))
        .await
        .expect("should succeed");

    assert!(
        resp.id.starts_with("chatcmpl-"),
        "id must start with chatcmpl-"
    );
    // model echoes the routing instruction from the request
    assert_eq!(
        resp.model, "test-model",
        "model must be preserved from request"
    );
    // Response messages contain exactly one assistant message from Provider
    let assistant_msg = resp
        .messages
        .iter()
        .find(|m| m.role == Role::Assistant && m.source == Source::Provider)
        .expect("response must contain an assistant Provider message");
    assert_eq!(assistant_msg.role, Role::Assistant);
    assert_eq!(
        resp.usage.total_tokens,
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    );
}

// ── Test: no user message ──────────────────────────────────────────────

#[tokio::test]
async fn test_no_user_message_returns_error() {
    let registry = single_model_registry(
        MockLlmProvider::single("irrelevant"),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    // A request with only a system message (no user message) should fail.
    let req = WeftRequest {
        messages: vec![WeftMessage {
            role: Role::System,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("system only".to_string())],
            delta: false,
            message_index: 0,
        }],
        routing: ModelRoutingInstruction::parse("test-model"),
        options: SamplingOptions::default(),
    };

    let result = engine.handle_request(req).await;
    assert!(result.is_err(), "expected error for missing user message");
}

// ── Test: rate limit error propagated ─────────────────────────────────

#[tokio::test]
async fn test_rate_limit_error_propagated() {
    let registry = single_model_registry(RateLimitedLlmProvider, "test-model", "claude-test");
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(
            result,
            Err(WeftError::RateLimited {
                retry_after_ms: 1000
            })
        ),
        "expected RateLimited, got: {:?}",
        result
    );
}

// ── Test: LLM error propagated ─────────────────────────────────────────

#[tokio::test]
async fn test_llm_error_propagated() {
    // Single-model registry: failing provider IS the default, so no fallback.
    let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(result, Err(WeftError::Llm(_))),
        "expected Llm error, got: {:?}",
        result
    );
}

// ── Test: list commands failure ────────────────────────────────────────

#[tokio::test]
async fn test_list_commands_failure_propagated() {
    let registry = single_model_registry(
        MockLlmProvider::single("irrelevant"),
        "test-model",
        "claude-test",
    );
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::with_list_failure(),
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(result, Err(WeftError::Command(_))),
        "expected Command error from list failure, got: {:?}",
        result
    );
}

// ── Test: empty LLM response is valid ─────────────────────────────────

#[tokio::test]
async fn test_empty_llm_response_is_valid() {
    let registry =
        single_model_registry(MockLlmProvider::single(""), "test-model", "claude-test");
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("empty LLM response must be valid");

    assert_eq!(resp_text(&resp), "");
}

// ── Test: usage propagated from last LLM call ──────────────────────────

#[tokio::test]
async fn test_usage_extracted_from_provider() {
    let registry =
        single_model_registry(MockLlmProvider::single("Done"), "test-model", "claude-test");
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed");

    // MockLlmProvider returns prompt=10, completion=5
    assert_eq!(resp.usage.prompt_tokens, 10);
    assert_eq!(resp.usage.completion_tokens, 5);
    assert_eq!(resp.usage.total_tokens, 15);
}

#[tokio::test]
async fn test_end_to_end_uses_provider_execute_path() {
    // End-to-end: request flows through Provider::execute() and produces correct response.
    // Verifies the full path from handle_request -> run_loop -> call_with_fallback ->
    // provider.execute(ProviderRequest::ChatCompletion) -> ProviderResponse::ChatCompletion.
    let provider = MockLlmProvider::single("Hello from provider");
    let registry = single_model_registry(provider, "test-model", "claude-test");

    let engine = make_engine(
        registry,
        MockRouter::with_model("test-model"),
        MockCommandRegistry::new(vec![]),
    );

    let resp = engine
        .handle_request(make_user_request("Hi there"))
        .await
        .expect("end-to-end should succeed");

    assert_eq!(
        resp_text(&resp),
        "Hello from provider",
        "response text must come from provider.execute()"
    );
    // Usage should be populated from the mock provider.
    assert_eq!(resp.usage.prompt_tokens, 10);
    assert_eq!(resp.usage.completion_tokens, 5);
}

#[tokio::test]
async fn test_no_eligible_models_error_message() {
    // Verify the error message format for NoEligibleModels.
    let err = WeftError::NoEligibleModels {
        capability: "chat_completions".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "no models configured with capability 'chat_completions'"
    );
}
