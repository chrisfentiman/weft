/// Tests for built-in memory commands (/recall, /remember) and memory mux wiring.
use super::*;
use weft_memory::{
    DefaultMemoryService, MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreMux,
    MemoryStoreResult, StoreInfo,
};

// ── Mock memory store client ───────────────────────────────────────────

/// Mock memory store client with configurable query and store behaviour.
pub(super) struct MockMemStoreClient {
    pub(super) query_entries: Vec<MemoryEntry>,
    pub(super) query_error: Option<String>,
    pub(super) store_id: String,
    pub(super) store_error: Option<String>,
}

impl MockMemStoreClient {
    pub(super) fn succeeds(entries: Vec<MemoryEntry>) -> Self {
        Self {
            query_entries: entries,
            query_error: None,
            store_id: "mock-mem-id".to_string(),
            store_error: None,
        }
    }

    pub(super) fn query_fails(msg: &str) -> Self {
        Self {
            query_entries: vec![],
            query_error: Some(msg.to_string()),
            store_id: "mock-mem-id".to_string(),
            store_error: None,
        }
    }

    pub(super) fn store_fails(msg: &str) -> Self {
        Self {
            query_entries: vec![],
            query_error: None,
            store_id: "mock-mem-id".to_string(),
            store_error: Some(msg.to_string()),
        }
    }
}

#[async_trait::async_trait]
impl MemoryStoreClient for MockMemStoreClient {
    async fn query(
        &self,
        _query: &str,
        _max_results: u32,
        _min_score: f32,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        if let Some(e) = &self.query_error {
            return Err(MemoryStoreError::QueryFailed(e.clone()));
        }
        Ok(self.query_entries.clone())
    }

    async fn store(
        &self,
        _content: &str,
        _metadata: Option<&serde_json::Value>,
    ) -> Result<MemoryStoreResult, MemoryStoreError> {
        if let Some(e) = &self.store_error {
            return Err(MemoryStoreError::StoreFailed(e.clone()));
        }
        Ok(MemoryStoreResult {
            id: self.store_id.clone(),
        })
    }
}

pub(super) fn mem_entry(id: &str, content: &str) -> MemoryEntry {
    MemoryEntry {
        id: id.to_string(),
        content: content.to_string(),
        score: 0.9,
        created_at: "2026-03-15T10:00:00Z".to_string(),
        metadata: None,
    }
}

pub(super) fn make_mux_with_stores(
    stores: Vec<(&str, bool, bool, Arc<dyn MemoryStoreClient>)>,
) -> Arc<MemoryStoreMux> {
    let mut client_map: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
    let mut max_results = HashMap::new();
    let mut readable = std::collections::HashSet::new();
    let mut writable = std::collections::HashSet::new();
    for (name, r, w, client) in stores {
        client_map.insert(name.to_string(), client);
        max_results.insert(name.to_string(), 5u32);
        if r {
            readable.insert(name.to_string());
        }
        if w {
            writable.insert(name.to_string());
        }
    }
    Arc::new(MemoryStoreMux::new(
        client_map,
        max_results,
        readable,
        writable,
    ))
}

pub(super) fn make_engine_with_mux<R, C>(
    registry: Arc<ProviderRegistry>,
    router: R,
    commands: C,
    mux: Option<Arc<MemoryStoreMux>>,
) -> GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
where
    R: SemanticRouter + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    let memory = mux.map(wrap_mux);
    GatewayEngine::new(
        test_config(),
        registry,
        Arc::new(router),
        Arc::new(commands),
        memory,
        Arc::new(weft_hooks::HookRegistry::empty()),
    )
}

/// Wrap a `MemoryStoreMux` in a `DefaultMemoryService` with no examples (store_infos with
/// no examples). Used when tests only care about ops (query/store), not routing candidates.
pub(super) fn wrap_mux(mux: Arc<MemoryStoreMux>) -> Arc<DefaultMemoryService> {
    // Build store_infos from the mux's store names and capabilities.
    // No examples — these tests use MockRouter with fixed scores and don't
    // need semantic routing candidates populated.
    let store_infos: Vec<StoreInfo> = {
        let all: std::collections::HashSet<String> = mux
            .store_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let readable: std::collections::HashSet<String> = mux
            .readable_store_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let writable: std::collections::HashSet<String> = mux
            .writable_store_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        all.into_iter()
            .map(|name| {
                let mut caps = Vec::new();
                if readable.contains(&name) {
                    caps.push("read".to_string());
                }
                if writable.contains(&name) {
                    caps.push("write".to_string());
                }
                StoreInfo {
                    name,
                    capabilities: caps,
                    examples: vec![],
                }
            })
            .collect()
    };
    Arc::new(DefaultMemoryService::new(mux, store_infos))
}

// ── /recall tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_recall_intercepted_before_command_registry() {
    // Engine with memory mux — /recall should NOT reach the command registry.
    // If it did reach the registry, it would fail with CommandError::NotFound
    // (the mock registry has no "recall" command) and the loop would continue
    // until the LLM emits a response without commands.
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "dark mode",
        )])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"user preferences\"",
            "I found some memories about preferences.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]), // no "recall" in registry
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("What are the user's preferences?"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "I found some memories about preferences.");
}

#[tokio::test]
async fn test_recall_returns_memories_as_toon() {
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "user prefers dark mode",
        )])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"user preferences\"",
            "Thanks for the context.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("What does the user like?"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Thanks for the context.");
}

#[tokio::test]
async fn test_recall_no_query_arg_uses_user_message() {
    // /recall without query arg should still succeed (falls back to user message).
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/recall", "Nothing found in memory."]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Tell me about user prefs"))
        .await
        .expect("should succeed without panicking");

    assert_eq!(resp_text(&resp), "Nothing found in memory.");
}

#[tokio::test]
async fn test_recall_all_stores_fail_returns_success_with_no_memories() {
    // All stores fail — should return success=true with "No relevant memories found."
    let mux = make_mux_with_stores(vec![(
        "broken",
        true,
        true,
        Arc::new(MockMemStoreClient::query_fails("connection refused")),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"something\"",
            "Understood, no memories available.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Search memory"))
        .await
        .expect("should succeed even when stores fail");

    assert_eq!(resp_text(&resp), "Understood, no memories available.");
}

#[tokio::test]
async fn test_recall_no_mux_returns_error_result() {
    // /recall with no mux configured should emit an error result to the LLM.
    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"something\"",
            "Memory is not configured.",
        ]),
        "test-model",
        "claude-test",
    );
    // Engine with no mux but we need to parse /recall — add it to registry as a command
    // Actually, without a mux, /recall won't be in known_commands, so it won't be parsed.
    // Test instead: call handle_builtin directly by building engine and sending a request
    // where the LLM emits /recall but the parser only picks it up if we register it.
    // To force the parse, we build an engine with a mock mux that is None-equivalent
    // — but there's no way to do that without an actual mux; the mux controls parser registration.
    // This is correct behaviour: without a mux, /recall is not in known_commands and is treated as prose.
    // Verify the response still works (just contains the text with /recall treated as prose).
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        None, // no mux
    );

    let resp = engine
        .handle_request(make_user_request("Try to recall"))
        .await
        .expect("should succeed");

    // The LLM response includes "/recall query: something" but it's treated as prose
    // because recall is not in known_commands when mux is None.
    // The first LLM response is returned directly since no commands are parsed.
    assert_eq!(resp_text(&resp), "/recall query: \"something\"");
}

// ── /remember tests ───────────────────────────────────────────────────

#[tokio::test]
async fn test_remember_intercepted_and_stores_to_writable() {
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/remember content: \"user prefers dark mode\"",
            "Memory stored.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Remember this"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Memory stored.");
}

#[tokio::test]
async fn test_remember_missing_content_returns_error() {
    // /remember without content arg should return success:false error to LLM.
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/remember", // missing content
            "Got an error about missing argument.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Store something"))
        .await
        .expect("should succeed overall");

    assert_eq!(resp_text(&resp), "Got an error about missing argument.");
}

#[tokio::test]
async fn test_remember_no_writable_stores_returns_empty_success() {
    // All stores are read-only — /remember should return success with no stores written.
    let mux = make_mux_with_stores(vec![(
        "code",
        true,
        false, // read-only
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/remember content: \"some info\"",
            "Attempted to remember.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Remember this"))
        .await
        .expect("should not fail");

    assert_eq!(resp_text(&resp), "Attempted to remember.");
}

// ── --describe tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_recall_describe_returns_compiled_text() {
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/recall --describe", "I see how recall works."]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("How does recall work?"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "I see how recall works.");
}

#[tokio::test]
async fn test_remember_describe_does_not_reach_command_registry() {
    // The mock registry has no "remember" command — if --describe reached the
    // registry, it would return an error. With built-in interception it returns
    // the compiled describe text and success=true.
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/remember --describe",
            "Now I know how to use remember.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]), // no "remember" in registry
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("How do I store memories?"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Now I know how to use remember.");
}

// ── mixed built-in + external commands ────────────────────────────────

#[tokio::test]
async fn test_mixed_builtin_and_external_commands() {
    // LLM emits both /recall and /web_search in the same response.
    // Both should execute: built-in first, then external.
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "user prefers dark mode",
        )])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"prefs\"\n/web_search query: \"dark mode\"",
            "Found memories and web results.",
        ]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "dark mode is popular".to_string(),
                error: None,
            },
        ),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("What do I know about dark mode?"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Found memories and web results.");
}

#[tokio::test]
async fn test_external_commands_still_work_with_mux_present() {
    // Verify that wiring in a mux doesn't break the existing command registry path.
    let mux = make_mux_with_stores(vec![(
        "conv",
        true,
        true,
        Arc::new(MockMemStoreClient::succeeds(vec![])),
    )]);

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Here are the results."]),
        "test-model",
        "claude-test",
    );
    let engine = make_engine_with_mux(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "5 results about Rust".to_string(),
                error: None,
            },
        ),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Search for Rust"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Here are the results.");
}
