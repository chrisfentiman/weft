/// Tests for hook lifecycle integration (Phase 4).
use super::*;
use weft_core::{HookEvent, WeftError};
use weft_hooks::types::HookMatcher;
use weft_hooks::{HookRegistry, RegisteredHook};
use weft_memory::DefaultMemoryService;

// ── Hook infrastructure helpers ────────────────────────────────────────

/// Build a `GatewayEngine` with a custom hook registry (Phase 4 tests).
fn make_engine_with_hooks<R, C>(
    registry: Arc<ProviderRegistry>,
    router: R,
    commands: C,
    hook_registry: weft_hooks::HookRegistry,
) -> GatewayEngine<
    weft_hooks::HookRegistry,
    R,
    weft_memory::NullMemoryService,
    ProviderRegistry,
    C,
>
where
    R: SemanticRouter + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    GatewayEngine::new(
        test_config(),
        registry,
        Arc::new(router),
        Arc::new(commands),
        None::<Arc<weft_memory::NullMemoryService>>,
        Arc::new(hook_registry),
    )
}

/// Build a `HookRegistry` with a single hook executor for one event.
fn hook_registry_with(
    event: HookEvent,
    executor: Box<dyn weft_hooks::executor::HookExecutor>,
    matcher: Option<&str>,
    priority: u32,
) -> weft_hooks::HookRegistry {
    let matcher = HookMatcher::new(matcher, 0).expect("valid matcher in test");
    let hook = RegisteredHook {
        event,
        matcher,
        executor,
        name: "test-hook".to_string(),
        priority,
    };
    let mut map = std::collections::HashMap::new();
    map.insert(event, vec![hook]);
    HookRegistry::from_registered(map)
}

// ── Phase 4: inline hook executor helpers ─────────────────────────────

/// A hook executor that always returns a fixed response.
struct FixedHookExecutor(weft_hooks::types::HookResponse);

#[async_trait]
impl weft_hooks::executor::HookExecutor for FixedHookExecutor {
    async fn execute(&self, _payload: &serde_json::Value) -> weft_hooks::types::HookResponse {
        self.0.clone()
    }
}

/// A hook executor that records every payload it receives, then returns Allow.
struct RecordingHookExecutor {
    recorded: std::sync::Mutex<Vec<serde_json::Value>>,
}

impl RecordingHookExecutor {
    fn new() -> Self {
        Self {
            recorded: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn recorded_payloads(&self) -> Vec<serde_json::Value> {
        self.recorded.lock().unwrap().clone()
    }
}

#[async_trait]
impl weft_hooks::executor::HookExecutor for RecordingHookExecutor {
    async fn execute(&self, payload: &serde_json::Value) -> weft_hooks::types::HookResponse {
        self.recorded.lock().unwrap().push(payload.clone());
        weft_hooks::types::HookResponse::allow()
    }
}

/// A hook executor that modifies the payload by merging in a JSON object.
struct ModifyHookExecutor(serde_json::Value);

#[async_trait]
impl weft_hooks::executor::HookExecutor for ModifyHookExecutor {
    async fn execute(&self, payload: &serde_json::Value) -> weft_hooks::types::HookResponse {
        // Merge self.0 fields into current payload.
        let mut merged = payload.clone();
        if let (Some(obj), Some(extra)) = (merged.as_object_mut(), self.0.as_object()) {
            for (k, v) in extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        weft_hooks::types::HookResponse {
            decision: weft_hooks::types::HookDecision::Modify,
            reason: None,
            modified: Some(merged),
            context: None,
        }
    }
}

/// Build a HookRegistry with multiple hooks for potentially different events.
fn hook_registry_multi(
    hooks: Vec<(
        HookEvent,
        Box<dyn weft_hooks::executor::HookExecutor>,
        Option<&'static str>,
        u32,
    )>,
) -> weft_hooks::HookRegistry {
    let mut map: std::collections::HashMap<HookEvent, Vec<RegisteredHook>> =
        std::collections::HashMap::new();
    for (i, (event, executor, matcher_pat, priority)) in hooks.into_iter().enumerate() {
        let matcher = HookMatcher::new(matcher_pat, i).expect("valid matcher");
        let hook = RegisteredHook {
            event,
            matcher,
            executor,
            name: format!("test-hook-{i}"),
            priority,
        };
        map.entry(event).or_default().push(hook);
    }
    HookRegistry::from_registered(map)
}

// ── Phase 4: RequestStart hook integration ─────────────────────────────

#[tokio::test]
async fn test_request_start_hook_blocks_returns_hook_blocked_error() {
    // RequestStart block — 403, no LLM call made.
    let registry = single_model_registry(
        MockLlmProvider::single("should not be called"),
        "test-model",
        "claude-test",
    );
    let hook_reg = hook_registry_with(
        HookEvent::RequestStart,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
            "auth failed",
        ))),
        None,
        100,
    );
    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(
            result,
            Err(WeftError::HookBlocked { ref event, ref reason, .. })
                if event == "RequestStart" && reason == "auth failed"
        ),
        "expected HookBlocked for RequestStart, got: {result:?}"
    );
}

#[tokio::test]
async fn test_request_start_hook_allows_request_proceeds() {
    // RequestStart allow — request continues normally.
    let registry = single_model_registry(
        MockLlmProvider::single("Response text"),
        "test-model",
        "claude-test",
    );
    let hook_reg = hook_registry_with(
        HookEvent::RequestStart,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::allow())),
        None,
        100,
    );
    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed when RequestStart hook allows");
    assert_eq!(resp_text(&resp), "Response text");
}

// ── Phase 4: PreRoute hook integration ─────────────────────────────────

#[tokio::test]
async fn test_pre_route_hook_blocks_model_domain_returns_hook_blocked() {
    // PreRoute block on model domain → hard block (403).
    let hook_reg = hook_registry_with(
        HookEvent::PreRoute,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
            "model blocked",
        ))),
        Some("model"),
        100,
    );

    let engine = GatewayEngine::new(
        multi_model_config(),
        two_model_registry(
            MockLlmProvider::single("should not be called"),
            MockLlmProvider::single("should not be called"),
        ),
        Arc::new(MockRouter::with_model("complex-model")),
        Arc::new(MockCommandRegistry::new(vec![])),
        None::<Arc<weft_memory::NullMemoryService>>,
        Arc::new(hook_reg),
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(
            result,
            Err(WeftError::HookBlocked { ref event, .. })
                if event == "PreRoute"
        ),
        "expected HookBlocked for PreRoute on model domain, got: {result:?}"
    );
}

#[tokio::test]
async fn test_pre_route_hook_no_matcher_fires_on_all_domains() {
    // A PreRoute hook with no matcher fires on every domain.
    // Verify by collecting the `domain` field from each payload received.
    let recorder = Arc::new(RecordingHookExecutor::new());
    let recorder_clone = Arc::clone(&recorder);

    struct ArcRecording(Arc<RecordingHookExecutor>);
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for ArcRecording {
        async fn execute(
            &self,
            payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            self.0.recorded.lock().unwrap().push(payload.clone());
            weft_hooks::types::HookResponse::allow()
        }
    }

    let hook_reg = hook_registry_with(
        HookEvent::PreRoute,
        Box::new(ArcRecording(recorder_clone)),
        None, // no matcher — fires on all domains
        100,
    );

    let engine = make_engine_with_hooks(
        single_model_registry(MockLlmProvider::single("Done"), "test-model", "claude-test"),
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search")]),
        hook_reg,
    );

    engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed");

    let payloads = recorder.recorded_payloads();
    // There must be at least one PreRoute firing (commands domain is always present).
    assert!(
        !payloads.is_empty(),
        "expected PreRoute hook to fire at least once"
    );
    // Extract domain names from payloads.
    let domains: Vec<&str> = payloads
        .iter()
        .filter_map(|p| p.get("domain").and_then(|d| d.as_str()))
        .collect();
    assert!(
        domains.contains(&"commands"),
        "expected PreRoute to fire for commands domain, got: {domains:?}"
    );
}

// ── Phase 4: PostRoute hook integration ────────────────────────────────

#[tokio::test]
async fn test_post_route_hook_overrides_model_selection() {
    // PostRoute hook on model domain overrides `selected` → correct model used.
    // Router would pick complex-model, but PostRoute hook forces default-model.
    // The `selected` field in the model PostRoute payload is an array of model name strings.
    let hook_reg = hook_registry_with(
        HookEvent::PostRoute,
        Box::new(ModifyHookExecutor(serde_json::json!({
            "selected": ["default-model"]
        }))),
        Some("model"),
        100,
    );

    let engine = GatewayEngine::new(
        multi_model_config(),
        two_model_registry(
            // default-model returns the expected response.
            MockLlmProvider::single("final response"),
            // complex-model returns a sentinel — if this appears, the hook didn't work.
            MockLlmProvider::single("WRONG: complex-model was used"),
        ),
        Arc::new(MockRouter::with_model("complex-model")),
        Arc::new(MockCommandRegistry::new(vec![])),
        None::<Arc<weft_memory::NullMemoryService>>,
        Arc::new(hook_reg),
    );

    // If PostRoute correctly overrides to default-model, we get "final response".
    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed with PostRoute model override to default-model");
    assert_eq!(resp_text(&resp), "final response");
}

#[tokio::test]
async fn test_post_route_hook_overrides_model_to_invalid_falls_back_to_default() {
    // PostRoute hook overrides model to a name not in the registry → fallback to default.
    // `selected` is an array of model name strings (same format as the PostRoute payload).
    let hook_reg = hook_registry_with(
        HookEvent::PostRoute,
        Box::new(ModifyHookExecutor(serde_json::json!({
            "selected": ["nonexistent-model"]
        }))),
        Some("model"),
        100,
    );

    let engine = GatewayEngine::new(
        multi_model_config(),
        two_model_registry(
            MockLlmProvider::single("fallback response"),
            MockLlmProvider::single("should not be called"),
        ),
        Arc::new(MockRouter::with_model("complex-model")),
        Arc::new(MockCommandRegistry::new(vec![])),
        None::<Arc<weft_memory::NullMemoryService>>,
        Arc::new(hook_reg),
    );

    // Should succeed by falling back to default-model.
    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should fall back to default model on invalid PostRoute override");
    assert_eq!(resp_text(&resp), "fallback response");
}

// ── Phase 4: PreToolUse hook integration ───────────────────────────────

#[tokio::test]
async fn test_pre_tool_use_blocks_command_returns_failed_result() {
    // PreToolUse blocks web_search → command result is an error in conversation.
    // The LLM sees the error and provides a final response.
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search query: \"Rust\"",
            "Search blocked, answering from memory.",
        ]),
        "test-model",
        "claude-test",
    );

    let hook_reg = hook_registry_with(
        HookEvent::PreToolUse,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
            "permission denied",
        ))),
        Some("web_search"),
        100,
    );

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Search for Rust"))
        .await
        .expect("should succeed — block returns error result not 403");
    assert_eq!(resp_text(&resp), "Search blocked, answering from memory.");
}

#[tokio::test]
async fn test_pre_tool_use_blocks_one_command_other_executes() {
    // PreToolUse blocks web_search (matcher="web_search") but allows code_review.
    // Both are invoked in the same turn — only code_review executes.
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search query: \"Rust\"\n/code_review target: src",
            "Done.",
        ]),
        "test-model",
        "claude-test",
    );

    let hook_reg = hook_registry_with(
        HookEvent::PreToolUse,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
            "blocked",
        ))),
        Some("web_search"), // only blocks web_search
        100,
    );

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search"), ("code_review", "Review")])
            .with_execute_result(CommandResult {
                command_name: "code_review".to_string(),
                success: true,
                output: "review done".to_string(),
                error: None,
            }),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Do both"))
        .await
        .expect("should succeed");
    assert_eq!(resp_text(&resp), "Done.");
}

// ── Phase 4: PostToolUse hook integration ──────────────────────────────

#[tokio::test]
async fn test_post_tool_use_hook_modifies_output() {
    // PostToolUse hook modifies the command output — modified text visible to LLM.
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/web_search query: \"Rust\"",
            "Search result was: hooked output",
        ]),
        "test-model",
        "claude-test",
    );

    let hook_reg = hook_registry_with(
        HookEvent::PostToolUse,
        Box::new(ModifyHookExecutor(serde_json::json!({
            "output": "hooked output"
        }))),
        Some("web_search"),
        100,
    );

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "original output".to_string(),
                error: None,
            },
        ),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Search for Rust"))
        .await
        .expect("should succeed");
    assert_eq!(resp_text(&resp), "Search result was: hooked output");
}

// ── Phase 4: PreResponse hook integration ──────────────────────────────

#[tokio::test]
async fn test_pre_response_hook_blocks_triggers_regeneration() {
    // PreResponse blocks once → LLM regenerates; second response passes.
    // The mock LLM returns two responses in order.
    let registry = single_model_registry(
        MockLlmProvider::new(vec!["first response", "second response"]),
        "test-model",
        "claude-test",
    );

    // Block on first call, then allow (FixedHookExecutor always returns same response,
    // so we use a stateful executor that blocks once).
    struct BlockOnceExecutor {
        count: std::sync::Mutex<u32>,
    }
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for BlockOnceExecutor {
        async fn execute(
            &self,
            _payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            let mut count = self.count.lock().unwrap();
            *count += 1;
            if *count == 1 {
                weft_hooks::types::HookResponse::block("content policy")
            } else {
                weft_hooks::types::HookResponse::allow()
            }
        }
    }

    let hook_reg = hook_registry_with(
        HookEvent::PreResponse,
        Box::new(BlockOnceExecutor {
            count: std::sync::Mutex::new(0),
        }),
        None,
        100,
    );

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed after regeneration");
    // Second response should be returned (first was blocked and regenerated).
    assert_eq!(resp_text(&resp), "second response");
}

#[tokio::test]
async fn test_pre_response_hook_blocks_after_max_retries_returns_422() {
    // PreResponse blocks on every attempt → HTTP 422 after max retries exhausted.
    // test_config() sets max_pre_response_retries = 2.
    let registry = single_model_registry(
        // Provide enough responses for the retry loop + 1 extra.
        MockLlmProvider::new(vec!["bad response"; 10]),
        "test-model",
        "claude-test",
    );

    let hook_reg = hook_registry_with(
        HookEvent::PreResponse,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
            "always blocked",
        ))),
        None,
        100,
    );

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(
            result,
            Err(WeftError::HookBlockedAfterRetries { ref event, retries, .. })
                if event == "PreResponse" && retries == 2
        ),
        "expected HookBlockedAfterRetries after max retries, got: {result:?}"
    );
}

#[tokio::test]
async fn test_pre_response_hook_modifies_response_text() {
    // PreResponse Modify changes the response text → client sees modified text.
    let registry = single_model_registry(
        MockLlmProvider::single("original text"),
        "test-model",
        "claude-test",
    );

    let hook_reg = hook_registry_with(
        HookEvent::PreResponse,
        Box::new(ModifyHookExecutor(serde_json::json!({
            "text": "modified text"
        }))),
        None,
        100,
    );

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed");
    assert_eq!(resp_text(&resp), "modified text");
}

// ── Phase 4: RequestEnd hook integration ───────────────────────────────

#[tokio::test]
async fn test_request_end_hook_fires_after_response() {
    // RequestEnd fires after the response is built (verify via side-effect).
    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fired_clone = Arc::clone(&fired);

    struct FireFlagExecutor(Arc<std::sync::atomic::AtomicBool>);
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for FireFlagExecutor {
        async fn execute(
            &self,
            _payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            self.0.store(true, std::sync::atomic::Ordering::Release);
            weft_hooks::types::HookResponse::allow()
        }
    }

    let hook_reg = hook_registry_with(
        HookEvent::RequestEnd,
        Box::new(FireFlagExecutor(fired_clone)),
        None,
        100,
    );

    let engine = make_engine_with_hooks(
        single_model_registry(
            MockLlmProvider::single("Hello"),
            "test-model",
            "claude-test",
        ),
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed");
    assert_eq!(resp_text(&resp), "Hello");

    // Give the spawned RequestEnd task a moment to execute.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        fired.load(std::sync::atomic::Ordering::Acquire),
        "RequestEnd hook should have fired after response"
    );
}

#[tokio::test]
async fn test_request_end_semaphore_exhausted_drops_task_with_warning() {
    // Semaphore with 0 permits → RequestEnd task dropped (no panic, no hang).
    // Build a config with request_end_concurrency = 0.
    let config = Arc::new(WeftConfig {
        request_end_concurrency: 0,
        ..(*test_config()).clone()
    });

    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fired_clone = Arc::clone(&fired);

    struct FireFlagExecutor(Arc<std::sync::atomic::AtomicBool>);
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for FireFlagExecutor {
        async fn execute(
            &self,
            _payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            self.0.store(true, std::sync::atomic::Ordering::Release);
            weft_hooks::types::HookResponse::allow()
        }
    }

    let hook_reg = hook_registry_with(
        HookEvent::RequestEnd,
        Box::new(FireFlagExecutor(fired_clone)),
        None,
        100,
    );

    let engine = GatewayEngine::new(
        config,
        single_model_registry(
            MockLlmProvider::single("Hello"),
            "test-model",
            "claude-test",
        ),
        Arc::new(MockRouter::with_score(0.9)),
        Arc::new(MockCommandRegistry::new(vec![])),
        None::<Arc<weft_memory::NullMemoryService>>,
        Arc::new(hook_reg),
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed even with exhausted semaphore");
    assert_eq!(resp_text(&resp), "Hello");

    // Give time for any possible task scheduling.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // Task was dropped — hook should NOT have fired.
    assert!(
        !fired.load(std::sync::atomic::Ordering::Acquire),
        "RequestEnd hook should have been dropped due to semaphore exhaustion"
    );
}

// ── Phase 4: no-hooks unchanged behavior ───────────────────────────────

#[tokio::test]
async fn test_no_hooks_configured_behavior_unchanged() {
    // With no hooks, engine behaves identically to existing tests.
    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Results found."]),
        "test-model",
        "claude-test",
    );
    // Default make_engine uses HookRegistry::empty() — explicitly verify.
    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "Rust results".to_string(),
                error: None,
            },
        ),
    );

    let resp = engine
        .handle_request(make_user_request("Search for Rust"))
        .await
        .expect("should succeed with no hooks");
    assert_eq!(resp_text(&resp), "Results found.");
}

// ── Phase 4: hook priority ordering ────────────────────────────────────

#[tokio::test]
async fn test_hooks_execute_in_priority_order_lower_first() {
    // Two RequestStart hooks: priority 200 blocks, priority 50 allows.
    // Since lower fires first, priority 50 (allow) fires before priority 200 (block).
    // The block at priority 200 runs second → request is blocked.
    // This verifies that priority 200 hook's block is respected even after allow.
    //
    // Inverse test: priority 50 blocks, 200 allows → blocked at priority 50.
    let registry = single_model_registry(
        MockLlmProvider::single("should not be called"),
        "test-model",
        "claude-test",
    );

    // Two hooks: low priority blocks, high priority allows.
    // Since lower priority value fires first, the block hook (50) fires first → blocked.
    let hook_reg = hook_registry_multi(vec![
        (
            HookEvent::RequestStart,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "low priority block",
            ))),
            None,
            50, // fires first
        ),
        (
            HookEvent::RequestStart,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::allow())),
            None,
            200, // fires second (but chain already blocked)
        ),
    ]);

    let engine = make_engine_with_hooks(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(
            result,
            Err(WeftError::HookBlocked { ref reason, .. }) if reason == "low priority block"
        ),
        "lower priority hook should have fired first and blocked, got: {result:?}"
    );
}

// ── Phase 4: memory command hook integration ───────────────────────────

/// Build a `GatewayEngine` with a custom hook registry AND a memory mux.
fn make_engine_with_hooks_and_mux<R, C>(
    registry: Arc<ProviderRegistry>,
    router: R,
    commands: C,
    hook_registry: weft_hooks::HookRegistry,
    mux: Option<Arc<weft_memory::MemoryStoreMux>>,
) -> GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
where
    R: SemanticRouter + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    use super::memory_tests::wrap_mux;
    let memory = mux.map(wrap_mux);
    GatewayEngine::new(
        test_config(),
        registry,
        Arc::new(router),
        Arc::new(commands),
        memory,
        Arc::new(hook_registry),
    )
}

/// Build a minimal single-store memory mux with both read and write capability.
fn make_single_rw_mux(
    name: &str,
    client: Arc<dyn weft_memory::MemoryStoreClient>,
) -> Arc<weft_memory::MemoryStoreMux> {
    use super::memory_tests::make_mux_with_stores;
    make_mux_with_stores(vec![(name, true, true, client)])
}

#[tokio::test]
async fn test_pre_tool_use_blocks_recall_routing_hooks_not_fired() {
    // PreToolUse blocks /recall -> failed CommandResult returned, routing hooks
    // (PreRoute/PostRoute on memory domain) are NOT fired.
    //
    // Verified by: a PreRoute hook that panics if ever called. If routing hooks
    // were fired after the PreToolUse block, the test would panic.
    use super::memory_tests::{MockMemStoreClient, mem_entry};
    let mux = make_single_rw_mux(
        "conv",
        Arc::new(MockMemStoreClient::succeeds(vec![mem_entry("m1", "data")])),
    );

    struct PanicExecutor;
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for PanicExecutor {
        async fn execute(
            &self,
            _payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            panic!("PreRoute should not fire after PreToolUse block");
        }
    }

    let hook_reg = hook_registry_multi(vec![
        // PreToolUse blocks /recall.
        (
            HookEvent::PreToolUse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "recall blocked by policy",
            ))),
            Some("recall"),
            100,
        ),
        // PreRoute on memory domain — must NOT fire.
        (
            HookEvent::PreRoute,
            Box::new(PanicExecutor),
            Some("memory"),
            100,
        ),
    ]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/recall query: \"test\"", "No memory needed."]),
        "test-model",
        "claude-test",
    );

    let engine = make_engine_with_hooks_and_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
        Some(mux),
    );

    // The /recall should be blocked, but the engine should not panic and should
    // eventually return a response (after LLM sees the failed recall result).
    let resp = engine
        .handle_request(make_user_request("What do you know about me?"))
        .await
        .expect("engine should not error — blocked command is a failed CommandResult");

    assert_eq!(resp_text(&resp), "No memory needed.");
}

#[tokio::test]
async fn test_pre_tool_use_modifies_recall_arguments_used_in_routing() {
    // PreToolUse modifies /recall arguments -> the modified query is what reaches
    // the routing phase (and thus the memory store).
    //
    // Verified by: the mock mux client records which query text it receives.
    // We assert it matches the MODIFIED argument, not the original.
    use weft_memory::{MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreResult};
    let query_seen = Arc::new(std::sync::Mutex::new(String::new()));
    let query_seen_clone = Arc::clone(&query_seen);

    struct RecordingMemStoreClient {
        query_seen: Arc<std::sync::Mutex<String>>,
    }

    #[async_trait::async_trait]
    impl MemoryStoreClient for RecordingMemStoreClient {
        async fn query(
            &self,
            query: &str,
            _max_results: u32,
            _min_score: f32,
        ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
            *self.query_seen.lock().unwrap() = query.to_string();
            Ok(vec![])
        }

        async fn store(
            &self,
            _content: &str,
            _metadata: Option<&serde_json::Value>,
        ) -> Result<MemoryStoreResult, MemoryStoreError> {
            Ok(MemoryStoreResult {
                id: "mock".to_string(),
            })
        }
    }

    let mux = make_single_rw_mux(
        "conv",
        Arc::new(RecordingMemStoreClient {
            query_seen: query_seen_clone,
        }),
    );

    // PreToolUse hook: replace the "query" argument with a different value.
    let hook_reg = hook_registry_with(
        HookEvent::PreToolUse,
        Box::new(ModifyHookExecutor(serde_json::json!({
            "arguments": {"query": "modified query text"}
        }))),
        Some("recall"),
        100,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"original query\"",
            "Got the recall results.",
        ]),
        "test-model",
        "claude-test",
    );

    let engine = make_engine_with_hooks_and_mux(
        registry,
        MockRouter::with_score(0.9).with_memory_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
        Some(mux),
    );

    engine
        .handle_request(make_user_request("recall test"))
        .await
        .expect("should succeed");

    let seen = query_seen.lock().unwrap().clone();
    assert_eq!(
        seen, "modified query text",
        "routing should use the modified argument, not the original"
    );
}

#[tokio::test]
async fn test_pre_route_memory_blocks_recall_returns_failed_command_result() {
    // PreRoute hook with matcher "memory" blocks the recall routing phase ->
    // a failed CommandResult is returned to the LLM (feedback block).
    use super::memory_tests::MockMemStoreClient;
    let mux = make_single_rw_mux("conv", Arc::new(MockMemStoreClient::succeeds(vec![])));

    let hook_reg = hook_registry_with(
        HookEvent::PreRoute,
        Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
            "memory access denied",
        ))),
        Some("memory"),
        100,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"preferences\"",
            "Memory was blocked.",
        ]),
        "test-model",
        "claude-test",
    );

    let engine = make_engine_with_hooks_and_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
        Some(mux),
    );

    // The PreRoute block on memory results in a failed CommandResult, then
    // the LLM produces a final response after seeing the blocked recall.
    let resp = engine
        .handle_request(make_user_request("What do you know about me?"))
        .await
        .expect("engine should not error — blocked recall is a failed CommandResult");

    assert_eq!(resp_text(&resp), "Memory was blocked.");
}

#[tokio::test]
async fn test_post_route_memory_overrides_selected_stores_for_remember() {
    // PostRoute hook with matcher "memory" overrides selected stores for /remember.
    // There are two write-capable stores: "store-a" and "store-b".
    // Router would normally select "store-a" (score 0.9).
    // PostRoute overrides selected to ["store-b"].
    // Verified by: only "store-b"'s client records a store() call.
    use weft_memory::{MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreResult};
    use super::memory_tests::make_mux_with_stores;
    let store_a_stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let store_b_stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let store_a_clone = Arc::clone(&store_a_stored);
    let store_b_clone = Arc::clone(&store_b_stored);

    struct TrackingMemStoreClient(Arc<std::sync::atomic::AtomicBool>);

    #[async_trait::async_trait]
    impl MemoryStoreClient for TrackingMemStoreClient {
        async fn query(
            &self,
            _query: &str,
            _max_results: u32,
            _min_score: f32,
        ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
            Ok(vec![])
        }

        async fn store(
            &self,
            _content: &str,
            _metadata: Option<&serde_json::Value>,
        ) -> Result<MemoryStoreResult, MemoryStoreError> {
            self.0.store(true, std::sync::atomic::Ordering::Release);
            Ok(MemoryStoreResult {
                id: "tracked".to_string(),
            })
        }
    }

    let mux = make_mux_with_stores(vec![
        (
            "store-a",
            true,
            true,
            Arc::new(TrackingMemStoreClient(store_a_clone)),
        ),
        (
            "store-b",
            true,
            true,
            Arc::new(TrackingMemStoreClient(store_b_clone)),
        ),
    ]);

    // PostRoute on memory domain: override selected to ["store-b"].
    let hook_reg = hook_registry_with(
        HookEvent::PostRoute,
        Box::new(ModifyHookExecutor(serde_json::json!({
            "selected": ["store-b"]
        }))),
        Some("memory"),
        100,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/remember content: \"user prefers dark mode\"",
            "Stored the preference.",
        ]),
        "test-model",
        "claude-test",
    );

    // Router selects store-a (first candidate, score 0.9) — hook should override to store-b.
    let engine = make_engine_with_hooks_and_mux(
        registry,
        MockRouter::with_score(0.9).with_memory_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Remember that I prefer dark mode"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Stored the preference.");

    // Only store-b should have been written to.
    assert!(
        store_b_stored.load(std::sync::atomic::Ordering::Acquire),
        "store-b should have been written to after PostRoute override"
    );
    assert!(
        !store_a_stored.load(std::sync::atomic::Ordering::Acquire),
        "store-a should NOT have been written to (PostRoute overrode selection)"
    );
}

#[tokio::test]
async fn test_multiple_recall_invocations_each_fires_independent_hook_lifecycle() {
    // Multiple /recall invocations in the same LLM turn each fire their own
    // independent PreToolUse + PreRoute(memory) + PostRoute(memory) + PostToolUse.
    //
    // Verified by counting how many times PreToolUse is fired for "recall".
    use super::memory_tests::{MockMemStoreClient, mem_entry};
    let pre_tool_call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let count_clone = Arc::clone(&pre_tool_call_count);

    struct CountingExecutor(Arc<std::sync::atomic::AtomicU32>);
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for CountingExecutor {
        async fn execute(
            &self,
            _payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            weft_hooks::types::HookResponse::allow()
        }
    }

    let mux = make_single_rw_mux(
        "conv",
        Arc::new(MockMemStoreClient::succeeds(vec![mem_entry("m1", "data")])),
    );

    let matcher = HookMatcher::new(Some("recall"), 0).expect("valid matcher");
    let hook = RegisteredHook {
        event: HookEvent::PreToolUse,
        matcher,
        executor: Box::new(CountingExecutor(count_clone)),
        name: "counting-hook".to_string(),
        priority: 100,
    };
    let mut map = std::collections::HashMap::new();
    map.insert(HookEvent::PreToolUse, vec![hook]);
    let hook_reg = HookRegistry::from_registered(map);

    // LLM emits TWO /recall invocations in one turn.
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"query one\"\n/recall query: \"query two\"",
            "Done with both recalls.",
        ]),
        "test-model",
        "claude-test",
    );

    let engine = make_engine_with_hooks_and_mux(
        registry,
        MockRouter::with_score(0.9).with_memory_score(0.9),
        MockCommandRegistry::new(vec![]),
        hook_reg,
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Recall twice"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Done with both recalls.");

    // PreToolUse must have fired once per /recall invocation = 2 times.
    assert_eq!(
        pre_tool_call_count.load(std::sync::atomic::Ordering::Acquire),
        2,
        "PreToolUse should fire independently for each /recall invocation"
    );
}

#[tokio::test]
async fn test_post_route_modifies_scores_visible_to_subsequent_hooks() {
    // PostRoute hook modifies "scores" in the payload. A second PostRoute hook
    // that fires after the first should see the modified scores.
    //
    // Verified by: the second hook records what it received; we check that the
    // scores field reflects the first hook's modifications.
    let second_hook_payload = Arc::new(std::sync::Mutex::new(None::<serde_json::Value>));
    let capture_clone = Arc::clone(&second_hook_payload);

    struct CapturePayloadExecutor(Arc<std::sync::Mutex<Option<serde_json::Value>>>);
    #[async_trait]
    impl weft_hooks::executor::HookExecutor for CapturePayloadExecutor {
        async fn execute(
            &self,
            payload: &serde_json::Value,
        ) -> weft_hooks::types::HookResponse {
            *self.0.lock().unwrap() = Some(payload.clone());
            weft_hooks::types::HookResponse::allow()
        }
    }

    // First hook: modify scores to set all scores to 0.5.
    // Second hook: capture the payload after modification.
    let hook_reg = hook_registry_multi(vec![
        (
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "scores": [{"id": "test-model", "score": 0.5}]
            }))),
            Some("model"),
            50, // fires first
        ),
        (
            HookEvent::PostRoute,
            Box::new(CapturePayloadExecutor(capture_clone)),
            Some("model"),
            100, // fires second
        ),
    ]);

    let engine = GatewayEngine::new(
        multi_model_config(),
        two_model_registry(
            MockLlmProvider::single("Response"),
            MockLlmProvider::single("Response"),
        ),
        Arc::new(MockRouter::with_model("default-model")),
        Arc::new(MockCommandRegistry::new(vec![])),
        None::<Arc<weft_memory::NullMemoryService>>,
        Arc::new(hook_reg),
    );

    let resp = engine
        .handle_request(make_user_request("Hello"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Response");

    // The second PostRoute hook should have seen the scores modified by the first hook.
    let captured = second_hook_payload.lock().unwrap().clone();
    let captured = captured.expect("second hook should have been called");
    let scores = captured.get("scores").expect("scores field must exist");
    // The first hook set scores to [{"id": "test-model", "score": 0.5}].
    assert_eq!(
        scores,
        &serde_json::json!([{"id": "test-model", "score": 0.5}]),
        "second PostRoute hook should see scores modified by first hook"
    );
}
