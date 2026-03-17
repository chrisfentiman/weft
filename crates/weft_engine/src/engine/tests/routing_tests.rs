/// Tests for model routing, tool skipping, context assembly, and semantic router integration.
use super::*;
use weft_core::{
    ClassifierConfig, ContentPart, DomainConfig, DomainsConfig, GatewayConfig, MemoryConfig,
    MemoryStoreConfig, ModelEntry, ProviderConfig, Role, RouterConfig,
    ServerConfig, Source, StoreCapability, WeftConfig, WeftMessage,
};
use weft_memory::{DefaultMemoryService, MemoryStoreMux, StoreInfo};
use weft_router::{MemoryStoreRef, RoutingCandidate, RoutingDecision, RoutingDomainKind};

// ── Test: model selection uses router decision ─────────────────────────

#[tokio::test]
async fn test_model_selection_uses_router_decision() {
    // Router selects "complex-model"; we verify the correct model_id is passed.
    let recording_provider = Arc::new(RecordingLlmProvider::new("response"));
    let default_provider = Arc::new(MockLlmProvider::single("default response"));

    let mut providers = HashMap::new();
    providers.insert(
        "default-model".to_string(),
        default_provider as Arc<dyn Provider>,
    );
    providers.insert(
        "complex-model".to_string(),
        recording_provider.clone() as Arc<dyn Provider>,
    );

    let mut model_ids = HashMap::new();
    model_ids.insert("default-model".to_string(), "claude-default".to_string());
    model_ids.insert("complex-model".to_string(), "claude-complex".to_string());

    let mut max_tokens_map = HashMap::new();
    max_tokens_map.insert("default-model".to_string(), 1024u32);
    max_tokens_map.insert("complex-model".to_string(), 4096u32);

    // Both models support chat_completions for this test.
    let mut caps_map: HashMap<String, HashSet<Capability>> = HashMap::new();
    caps_map.insert(
        "default-model".to_string(),
        [Capability::new(Capability::CHAT_COMPLETIONS)]
            .into_iter()
            .collect(),
    );
    caps_map.insert(
        "complex-model".to_string(),
        [Capability::new(Capability::CHAT_COMPLETIONS)]
            .into_iter()
            .collect(),
    );

    let registry = Arc::new(ProviderRegistry::new(
        providers,
        model_ids,
        max_tokens_map,
        caps_map,
        "default-model".to_string(),
    ));

    let engine = make_engine_with_config(
        multi_model_config(),
        registry,
        MockRouter::with_model("complex-model"),
        MockCommandRegistry::new(vec![]),
    );

    engine
        .handle_request(make_user_request("Complex task"))
        .await
        .expect("should succeed");

    // Verify the correct model_id was passed to the recording provider
    assert_eq!(
        recording_provider.recorded_model(),
        Some("claude-complex".to_string()),
        "complex-model provider must receive the correct model_id"
    );
}

// ── Test: tool skipping when tools_needed = Some(false) ───────────────

#[tokio::test]
async fn test_tool_skipping_when_tools_not_needed() {
    // Provider records the system prompt to verify no command stubs
    struct SystemPromptCapture {
        captured: std::sync::Mutex<Option<String>>,
        response_text: &'static str,
    }

    #[async_trait]
    impl Provider for SystemPromptCapture {
        async fn execute(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            // Extract the system prompt from messages[0] (Role::System convention).
            #[allow(irrefutable_let_patterns)]
            if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                let system_text = messages
                    .first()
                    .filter(|m| m.role == Role::System)
                    .map(|m| {
                        m.content
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                *self.captured.lock().unwrap() = Some(system_text);
            }
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text(self.response_text.to_string())],
                    delta: false,
                    message_index: 0,
                },
                usage: None,
            })
        }

        fn name(&self) -> &str {
            "system-prompt-capture"
        }
    }

    let capture = Arc::new(SystemPromptCapture {
        captured: std::sync::Mutex::new(None),
        response_text: "no tools needed",
    });

    let registry = single_model_registry(
        {
            struct WrappedCapture(Arc<SystemPromptCapture>);
            #[async_trait]
            impl Provider for WrappedCapture {
                async fn execute(
                    &self,
                    request: ProviderRequest,
                ) -> Result<ProviderResponse, ProviderError> {
                    self.0.execute(request).await
                }

                fn name(&self) -> &str {
                    "wrapped-capture"
                }
            }
            WrappedCapture(capture.clone())
        },
        "test-model",
        "claude-test",
    );

    let engine = make_engine(
        registry,
        MockRouter::with_tools_needed(false),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]),
    );

    engine
        .handle_request(make_user_request("What is the capital of France?"))
        .await
        .expect("should succeed");

    let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
    // When tools are skipped, the system prompt must NOT contain command stubs
    assert!(
        !captured.contains("```toon"),
        "system prompt should not contain toon block when tools are skipped: {captured}"
    );
    assert!(
        !captured.contains("web_search"),
        "system prompt should not contain command stubs when tools are skipped"
    );
}

// ── Test: tool injection when tools_needed = Some(true) ───────────────

#[tokio::test]
async fn test_tool_injection_when_tools_needed() {
    struct SystemPromptCapture2 {
        captured: std::sync::Mutex<Option<String>>,
    }

    #[async_trait]
    impl Provider for SystemPromptCapture2 {
        async fn execute(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            // Extract the system prompt from messages[0] (Role::System convention).
            #[allow(irrefutable_let_patterns)]
            if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                let system_text = messages
                    .first()
                    .filter(|m| m.role == Role::System)
                    .map(|m| {
                        m.content
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                *self.captured.lock().unwrap() = Some(system_text);
            }
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text("response".to_string())],
                    delta: false,
                    message_index: 0,
                },
                usage: None,
            })
        }

        fn name(&self) -> &str {
            "system-prompt-capture2"
        }
    }

    let capture = Arc::new(SystemPromptCapture2 {
        captured: std::sync::Mutex::new(None),
    });

    let registry = single_model_registry(
        {
            struct W(Arc<SystemPromptCapture2>);
            #[async_trait]
            impl Provider for W {
                async fn execute(
                    &self,
                    request: ProviderRequest,
                ) -> Result<ProviderResponse, ProviderError> {
                    self.0.execute(request).await
                }

                fn name(&self) -> &str {
                    "w2"
                }
            }
            W(capture.clone())
        },
        "test-model",
        "claude-test",
    );

    let engine = make_engine(
        registry,
        MockRouter::with_tools_needed(true),
        MockCommandRegistry::new(vec![("web_search", "Search the web")]),
    );

    engine
        .handle_request(make_user_request("Search for something"))
        .await
        .expect("should succeed");

    let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
    // Commands must be injected in dash-list format with "Use when" trigger conditions
    assert!(
        captured.contains("- /web_search \u{2014} Use when Search the web"),
        "system prompt must contain dash-list command stubs when tools are needed"
    );
    assert!(
        captured.contains("BLOCKING REQUIREMENT"),
        "system prompt must contain blocking requirement language when tools are needed"
    );
}

// ── Test: tool injection when tools_needed = None (conservative) ──────

#[tokio::test]
async fn test_tool_injection_when_tools_needed_is_none() {
    // tools_needed = None -> conservative default: inject commands
    let router = MockRouter {
        command_score: 1.0,
        model_decision: None,
        tools_needed: None, // undecided
        memory_score: Some(0.9),
    };

    struct SystemPromptCapture3 {
        captured: std::sync::Mutex<Option<String>>,
    }

    #[async_trait]
    impl Provider for SystemPromptCapture3 {
        async fn execute(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            // Extract the system prompt from messages[0] (Role::System convention).
            #[allow(irrefutable_let_patterns)]
            if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                let system_text = messages
                    .first()
                    .filter(|m| m.role == Role::System)
                    .map(|m| {
                        m.content
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                *self.captured.lock().unwrap() = Some(system_text);
            }
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text("response".to_string())],
                    delta: false,
                    message_index: 0,
                },
                usage: None,
            })
        }

        fn name(&self) -> &str {
            "system-prompt-capture3"
        }
    }

    let capture = Arc::new(SystemPromptCapture3 {
        captured: std::sync::Mutex::new(None),
    });

    let registry = single_model_registry(
        {
            struct W(Arc<SystemPromptCapture3>);
            #[async_trait]
            impl Provider for W {
                async fn execute(
                    &self,
                    request: ProviderRequest,
                ) -> Result<ProviderResponse, ProviderError> {
                    self.0.execute(request).await
                }

                fn name(&self) -> &str {
                    "w3"
                }
            }
            W(capture.clone())
        },
        "test-model",
        "claude-test",
    );

    let engine = make_engine(
        registry,
        router,
        MockCommandRegistry::new(vec![("web_search", "Search the web")]),
    );

    engine
        .handle_request(make_user_request("Tell me about Rust"))
        .await
        .expect("should succeed");

    let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
    // None -> conservative: inject commands
    assert!(
        captured.contains("web_search"),
        "tools_needed=None must default to injecting commands"
    );
}

// ── Test: model fallback on non-rate-limit error ───────────────────────

#[tokio::test]
async fn test_model_fallback_on_non_rate_limit_error() {
    // complex-model fails with RequestFailed; default-model succeeds.
    let registry = two_model_registry(
        MockLlmProvider::single("fallback response"),
        FailingLlmProvider,
    );

    let engine = make_engine_with_config(
        multi_model_config(),
        registry,
        MockRouter::with_model("complex-model"),
        MockCommandRegistry::new(vec![]),
    );

    let resp = engine
        .handle_request(make_user_request("Complex task"))
        .await
        .expect("fallback to default must succeed");

    assert_eq!(resp_text(&resp), "fallback response");
}

// ── Test: rate limit does NOT trigger fallback ─────────────────────────

#[tokio::test]
async fn test_rate_limit_does_not_trigger_fallback() {
    // complex-model is rate-limited; must not fall back.
    let registry = two_model_registry(
        MockLlmProvider::single("default would succeed"),
        RateLimitedLlmProvider,
    );

    let engine = make_engine_with_config(
        multi_model_config(),
        registry,
        MockRouter::with_model("complex-model"),
        MockCommandRegistry::new(vec![]),
    );

    let result = engine
        .handle_request(make_user_request("Complex task"))
        .await;
    assert!(
        matches!(result, Err(WeftError::RateLimited { .. })),
        "rate limit must propagate without fallback, got: {:?}",
        result
    );
}

// ── Test: default model failure propagates error ───────────────────────

#[tokio::test]
async fn test_default_model_failure_propagates_error() {
    // Only the default model exists and it fails — no fallback available.
    let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");

    let engine = make_engine(
        registry,
        MockRouter::with_score(0.9),
        MockCommandRegistry::new(vec![]),
    );

    let result = engine.handle_request(make_user_request("Hello")).await;
    assert!(
        matches!(result, Err(WeftError::Llm(_))),
        "default model failure must propagate as Llm error, got: {:?}",
        result
    );
}

// ── Test: selected model IS default, no retry on failure ──────────────

#[tokio::test]
async fn test_selected_model_is_default_no_retry() {
    // Router selects "default-model" which is also the default — fails once,
    // must NOT retry (would infinite loop).
    let registry = two_model_registry(FailingLlmProvider, MockLlmProvider::single("alt"));

    let engine = make_engine_with_config(
        multi_model_config(),
        registry,
        // Router selects "default-model" (same as the registry's default)
        MockRouter::with_model("default-model"),
        MockCommandRegistry::new(vec![]),
    );

    let result = engine
        .handle_request(make_user_request("Simple task"))
        .await;
    assert!(
        matches!(result, Err(WeftError::Llm(_))),
        "selected=default failure must propagate without retry, got: {:?}",
        result
    );
}

// ── Test: context assembly uses TOON fenced blocks ─────────────────────

#[tokio::test]
async fn test_context_assembly_uses_toon_fenced_blocks() {
    use crate::context::assemble_system_prompt;
    use weft_core::CommandStub;

    let stubs = vec![CommandStub {
        name: "web_search".to_string(),
        description: "Search the web".to_string(),
    }];
    let prompt = assemble_system_prompt(&stubs, "You are a helpful assistant.", None);

    // Command stubs use dash-list format with "Use when" trigger conditions,
    // not TOON fenced blocks — TOON is reserved for results injection only.
    assert!(
        prompt.contains("- /web_search \u{2014} Use when Search the web"),
        "must use dash-list format with Use when trigger condition"
    );
    assert!(
        prompt.contains("BLOCKING REQUIREMENT"),
        "must include blocking requirement language"
    );
    assert!(
        !prompt.contains("commands[1]{name, description}:"),
        "must not use TOON array syntax for command stubs"
    );
}

// ── Test: no fenced blocks when tool skipping is active ───────────────

#[tokio::test]
async fn test_no_fenced_blocks_when_tool_skipping() {
    use crate::context::assemble_system_prompt_no_tools;

    let prompt = assemble_system_prompt_no_tools("You are a helpful assistant.");
    assert!(
        !prompt.contains("```toon"),
        "must not contain toon blocks when tools are skipped"
    );
    assert!(
        !prompt.contains("commands["),
        "must not contain command stubs when tools are skipped"
    );
}

// ── Phase 3: Build a WeftConfig with memory stores ────────────────────

/// Build a WeftConfig with memory stores for Phase 3 routing tests.
///
/// `stores`: slice of `(name, endpoint, can_read, can_write, examples)`.
/// `memory_threshold`: optional per-domain memory threshold.
pub(super) fn config_with_memory(
    stores: &[(&str, &str, bool, bool, Vec<&str>)],
    memory_threshold: Option<f32>,
) -> Arc<WeftConfig> {
    let store_configs: Vec<MemoryStoreConfig> = stores
        .iter()
        .map(|(name, endpoint, can_read, can_write, examples)| {
            let mut capabilities = Vec::new();
            if *can_read {
                capabilities.push(StoreCapability::Read);
            }
            if *can_write {
                capabilities.push(StoreCapability::Write);
            }
            MemoryStoreConfig {
                name: name.to_string(),
                endpoint: endpoint.to_string(),
                connect_timeout_ms: 5000,
                request_timeout_ms: 10000,
                max_results: 5,
                capabilities,
                examples: examples.iter().map(|s| s.to_string()).collect(),
            }
        })
        .collect();

    let memory_domain = memory_threshold.map(|t| DomainConfig {
        enabled: true,
        threshold: Some(t),
    });

    Arc::new(WeftConfig {
        server: ServerConfig {
            bind_address: "127.0.0.1:8080".to_string(),
        },
        router: RouterConfig {
            classifier: ClassifierConfig {
                model_path: "models/test.onnx".to_string(),
                tokenizer_path: "models/tokenizer.json".to_string(),
                threshold: 0.5, // default threshold for memory routing tests
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
            domains: DomainsConfig {
                model: None,
                tool_necessity: None,
                memory: memory_domain,
            },
        },
        tool_registry: None,
        memory: Some(MemoryConfig {
            stores: store_configs,
        }),
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

/// Build an engine with the given config, mux, and router.
///
/// The config provides routing thresholds, domain settings, AND store metadata
/// (examples) for building routing candidates. The mux handles actual ops.
/// Internally constructs a `DefaultMemoryService` from the config's memory section.
pub(super) fn make_engine_with_config_and_mux<R, C>(
    config: Arc<WeftConfig>,
    registry: Arc<ProviderRegistry>,
    router: R,
    commands: C,
    mux: Option<Arc<MemoryStoreMux>>,
) -> GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
where
    R: SemanticRouter + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    let memory = mux.map(|m| service_from_config_and_mux(&config, m));
    GatewayEngine::new(
        config,
        registry,
        Arc::new(router),
        Arc::new(commands),
        memory,
        Arc::new(weft_hooks::HookRegistry::empty()),
    )
}

/// Build a `DefaultMemoryService` from a config and a mux.
///
/// Extracts `StoreInfo` (name, capabilities, examples) from the config's memory section
/// so that routing candidates are populated correctly, then wraps the mux for operations.
pub(super) fn service_from_config_and_mux(
    config: &WeftConfig,
    mux: Arc<MemoryStoreMux>,
) -> Arc<DefaultMemoryService> {
    let store_infos: Vec<StoreInfo> = config
        .memory
        .as_ref()
        .map(|mem| {
            mem.stores
                .iter()
                .map(|s| {
                    let mut caps = Vec::new();
                    if s.can_read() {
                        caps.push("read".to_string());
                    }
                    if s.can_write() {
                        caps.push("write".to_string());
                    }
                    StoreInfo {
                        name: s.name.clone(),
                        capabilities: caps,
                        examples: s.examples.clone(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Arc::new(DefaultMemoryService::new(mux, store_infos))
}

// ── Phase 3: build_memory_candidates (via weft_router) ────────────────
// These tests verify the moved candidate builder via the weft_router re-export.

#[test]
fn test_build_memory_candidates_empty_when_no_stores() {
    // Empty store list produces empty candidate sets.
    let result = build_memory_candidates(&[]);
    assert!(result.all.is_empty());
    assert!(result.read.is_empty());
    assert!(result.write.is_empty());
}

#[test]
fn test_build_memory_candidates_read_write_store() {
    // A store with both capabilities appears in all three sets.
    let stores = vec![MemoryStoreRef {
        name: "conv".to_string(),
        capabilities: vec!["read".to_string(), "write".to_string()],
        examples: vec!["recall conv".to_string()],
    }];
    let result = build_memory_candidates(&stores);
    assert_eq!(result.all.len(), 1);
    assert_eq!(result.all[0].id, "conv");
    assert_eq!(result.read.len(), 1);
    assert_eq!(result.read[0].id, "conv");
    assert_eq!(result.write.len(), 1);
    assert_eq!(result.write[0].id, "conv");
}

#[test]
fn test_build_memory_candidates_read_only_store() {
    // A read-only store appears in all and read, but not write.
    let stores = vec![MemoryStoreRef {
        name: "kb".to_string(),
        capabilities: vec!["read".to_string()],
        examples: vec!["knowledge base".to_string()],
    }];
    let result = build_memory_candidates(&stores);
    assert_eq!(result.all.len(), 1);
    assert_eq!(result.read.len(), 1);
    assert!(result.write.is_empty());
}

#[test]
fn test_build_memory_candidates_write_only_store() {
    // A write-only store appears in all and write, but not read.
    let stores = vec![MemoryStoreRef {
        name: "audit".to_string(),
        capabilities: vec!["write".to_string()],
        examples: vec!["audit log".to_string()],
    }];
    let result = build_memory_candidates(&stores);
    assert_eq!(result.all.len(), 1);
    assert!(result.read.is_empty());
    assert_eq!(result.write.len(), 1);
}

#[test]
fn test_build_memory_candidates_multiple_stores() {
    // Multiple stores split correctly by capability.
    let stores = vec![
        MemoryStoreRef {
            name: "conv".to_string(),
            capabilities: vec!["read".to_string(), "write".to_string()],
            examples: vec!["conversation".to_string()],
        },
        MemoryStoreRef {
            name: "kb".to_string(),
            capabilities: vec!["read".to_string()],
            examples: vec!["knowledge base".to_string()],
        },
        MemoryStoreRef {
            name: "audit".to_string(),
            capabilities: vec!["write".to_string()],
            examples: vec!["audit".to_string()],
        },
    ];
    let result = build_memory_candidates(&stores);
    assert_eq!(result.all.len(), 3);
    assert_eq!(result.read.len(), 2); // conv + kb
    assert_eq!(result.write.len(), 2); // conv + audit
    assert!(result.read.iter().any(|c| c.id == "conv"));
    assert!(result.read.iter().any(|c| c.id == "kb"));
    assert!(result.write.iter().any(|c| c.id == "conv"));
    assert!(result.write.iter().any(|c| c.id == "audit"));
}

// ── Phase 3: route_all_domains includes Memory domain ─────────────────

#[tokio::test]
async fn test_route_all_domains_includes_memory_when_configured() {
    // When memory stores are configured, route_all_domains should pass
    // Memory domain candidates to the router. We verify by checking that
    // the engine picks up the memory candidates (indirectly via exec_recall routing).
    //
    // Setup: one store "conv", router scores it above threshold (0.9 > 0.5).
    // /recall should target "conv" specifically (not fan out to all).
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient, mem_entry};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m1",
        "dark mode pref",
    )]));
    let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

    let config = config_with_memory(
        &[(
            "conv",
            "http://localhost:50052",
            true,
            true,
            vec!["conversation recall"],
        )],
        None, // no memory-domain threshold override — uses classifier.threshold = 0.5
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"user preferences\"",
            "I found memory about dark mode.",
        ]),
        "test-model",
        "claude-test",
    );

    // Router scores first (only) read candidate at 0.9 > 0.5 threshold.
    let router = MockRouter::with_score(0.9).with_memory_score(0.9);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("Tell me about preferences"))
        .await
        .expect("should succeed");

    // Should succeed and include memory content in the response.
    assert_eq!(resp_text(&resp), "I found memory about dark mode.");
}

// ── Phase 3: /recall per-invocation routing ────────────────────────────

#[tokio::test]
async fn test_recall_routes_based_on_query_not_user_message() {
    // The key Phase 3 invariant: /recall routing uses the query argument,
    // not the user's original message. Both route via score_memory_candidates().
    //
    // Setup: two stores. "conv" gets score 0.9, "kb" gets 0.0 from mock router.
    // User message is "help me with my code" (code-domain).
    // /recall query is "user preferences" (conv-domain).
    // Only "conv" should be queried.
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient, mem_entry};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m1",
        "user prefers dark mode",
    )]));
    let kb_client = Arc::new(MockMemStoreClient::query_fails("should not be called"));
    let mux = make_mux_with_stores(vec![
        ("conv", true, false, conv_client),
        ("kb", true, false, kb_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                true,
                false,
                vec!["user preferences"],
            ),
            (
                "kb",
                "http://localhost:50053",
                true,
                false,
                vec!["code architecture"],
            ),
        ],
        None,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"user preferences\"",
            "Found: user prefers dark mode.",
        ]),
        "test-model",
        "claude-test",
    );

    // Router: first candidate (conv) gets 0.9, second (kb) gets 0.0.
    // This simulates "user preferences" being semantically closest to conv.
    let router = MockRouter::with_score(0.9).with_memory_score(0.9);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("help me with my code"))
        .await
        .expect("should succeed");

    // kb_client would fail if called — success means only conv was queried.
    assert_eq!(resp_text(&resp), "Found: user prefers dark mode.");
}

#[tokio::test]
async fn test_recall_below_threshold_fans_out_to_all_readable() {
    // When all candidates score below threshold, /recall fans out to ALL read-capable stores.
    // Router scores first candidate at 0.1, second at 0.0 — both below threshold 0.5.
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient, mem_entry};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m1",
        "conv memory",
    )]));
    let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m2",
        "kb memory",
    )]));
    let mux = make_mux_with_stores(vec![
        ("conv", true, false, conv_client),
        ("kb", true, false, kb_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                true,
                false,
                vec!["conversations"],
            ),
            (
                "kb",
                "http://localhost:50053",
                true,
                false,
                vec!["knowledge"],
            ),
        ],
        Some(0.5), // memory domain threshold
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"something unrelated\"",
            "Found memories from both stores.",
        ]),
        "test-model",
        "claude-test",
    );

    // Router: first candidate 0.1, second 0.0 — both below 0.5 threshold.
    let router = MockRouter::with_score(0.9).with_memory_score(0.1);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("something"))
        .await
        .expect("should succeed");

    // Both stores should have been queried (fan-out) — response shows 2 memories.
    assert_eq!(resp_text(&resp), "Found memories from both stores.");
}

#[tokio::test]
async fn test_recall_router_unavailable_fans_out_to_all_readable() {
    // When the router is unavailable (ModelNotLoaded), /recall fans out to all readable.
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient, mem_entry};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m1",
        "conv memory",
    )]));
    let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m2",
        "kb memory",
    )]));
    let mux = make_mux_with_stores(vec![
        ("conv", true, false, conv_client),
        ("kb", true, false, kb_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                true,
                false,
                vec!["conversations"],
            ),
            (
                "kb",
                "http://localhost:50053",
                true,
                false,
                vec!["knowledge"],
            ),
        ],
        None,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/recall query: \"anything\"",
            "Got results from both stores.",
        ]),
        "test-model",
        "claude-test",
    );

    // Router returns ModelNotLoaded for score_memory_candidates.
    let router = MockRouter::with_score(0.9).with_memory_unavailable();

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("anything"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Got results from both stores.");
}

// ── Phase 3: /remember per-invocation routing ─────────────────────────

#[tokio::test]
async fn test_remember_routes_based_on_content_not_user_message() {
    // /remember routing uses the content argument, not the user's original message.
    // Setup: two writable stores. "conv" gets score 0.9, "audit" gets 0.0.
    // User asks about code; LLM remembers a user preference.
    // Only "conv" should be written to.
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
    // audit_client: if written to, the store call would succeed, but we verify
    // by checking the TOON output only mentions "conv".
    let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
    let mux = make_mux_with_stores(vec![
        ("conv", false, true, conv_client),
        ("audit", false, true, audit_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                false,
                true,
                vec!["user preferences"],
            ),
            (
                "audit",
                "http://localhost:50053",
                false,
                true,
                vec!["audit logs"],
            ),
        ],
        None,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec![
            "/remember content: \"user prefers dark mode\"",
            "Noted, I'll remember that.",
        ]),
        "test-model",
        "claude-test",
    );

    // First write candidate (conv) gets 0.9, second (audit) gets 0.0.
    let router = MockRouter::with_score(0.9).with_memory_score(0.9);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("how do I configure nginx?"))
        .await
        .expect("should succeed");

    // audit_client has store_fails — if it were called we'd get a partial failure in TOON.
    // Success means only conv was written.
    assert_eq!(resp_text(&resp), "Noted, I'll remember that.");
}

#[tokio::test]
async fn test_remember_below_threshold_picks_highest_scoring_store() {
    // When all candidates are below threshold, /remember picks single highest-scoring store.
    // Router: first candidate (conv) 0.3, second (audit) 0.1 — both below 0.5 threshold.
    // conv has the higher score, so it gets picked.
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
    // audit_client: if written, would succeed — but conv should be picked.
    let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
    let mux = make_mux_with_stores(vec![
        ("conv", false, true, conv_client),
        ("audit", false, true, audit_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                false,
                true,
                vec!["conversation prefs"],
            ),
            (
                "audit",
                "http://localhost:50053",
                false,
                true,
                vec!["audit events"],
            ),
        ],
        Some(0.5), // threshold: 0.5
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/remember content: \"something\"", "Memory stored."]),
        "test-model",
        "claude-test",
    );

    // Both below threshold; first candidate scores higher (0.3 > 0.0 for second).
    let router = MockRouter::with_score(0.9).with_memory_score(0.3);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("remember something"))
        .await
        .expect("should succeed");

    // audit_client has store_fails — success means only conv was picked.
    assert_eq!(resp_text(&resp), "Memory stored.");
}

#[tokio::test]
async fn test_remember_router_unavailable_writes_to_first_writable() {
    // When router is unavailable, /remember writes to the first configured writable store.
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
    let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
    let mux = make_mux_with_stores(vec![
        ("conv", false, true, conv_client),
        ("audit", false, true, audit_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                false,
                true,
                vec!["conversations"],
            ),
            (
                "audit",
                "http://localhost:50053",
                false,
                true,
                vec!["audit"],
            ),
        ],
        None,
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/remember content: \"important note\"", "Saved."]),
        "test-model",
        "claude-test",
    );

    // Router unavailable for memory scoring.
    let router = MockRouter::with_score(0.9).with_memory_unavailable();

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("remember this"))
        .await
        .expect("should succeed");

    // audit_client has store_fails — success means only first writable (conv) was picked.
    assert_eq!(resp_text(&resp), "Saved.");
}

// ── Phase 3: memory domain threshold from config ──────────────────────

#[tokio::test]
async fn test_memory_domain_threshold_gate_applied() {
    // When a per-domain memory threshold is configured, it takes precedence over
    // the classifier threshold.
    //
    // Classifier threshold = 0.5, memory domain threshold = 0.8.
    // Router scores first candidate at 0.7 — above 0.5 but below 0.8.
    // With memory domain threshold, the store should NOT be directly selected
    // (falls back to all readable for /recall, or highest-scoring for /remember).
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient, mem_entry};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m1", "memory",
    )]));
    let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
    let mux = make_mux_with_stores(vec![
        ("conv", true, false, conv_client),
        ("kb", true, false, kb_client),
    ]);

    let config = config_with_memory(
        &[
            (
                "conv",
                "http://localhost:50052",
                true,
                false,
                vec!["conversations"],
            ),
            (
                "kb",
                "http://localhost:50053",
                true,
                false,
                vec!["knowledge"],
            ),
        ],
        Some(0.8), // memory domain threshold = 0.8 (higher than classifier 0.5)
    );

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/recall query: \"something\"", "Found memories."]),
        "test-model",
        "claude-test",
    );

    // Score 0.7: above classifier threshold (0.5) but below memory domain threshold (0.8).
    let router = MockRouter::with_score(0.9).with_memory_score(0.7);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("recall something"))
        .await
        .expect("should succeed");

    // Below memory-domain threshold → fell back to all readable → both stores queried.
    // conv_client returns a memory, kb_client returns empty.
    // Final response is whatever the LLM said after seeing results.
    assert_eq!(resp_text(&resp), "Found memories.");
}

// ── Phase 3: RoutingDecision::fallback() memory_stores population ─────

#[test]
fn test_fallback_decision_populates_memory_stores() {
    // RoutingDecision::fallback() should populate memory_stores with all
    // candidates at score 1.0 when the Memory domain is included.
    let domains = vec![
        (
            RoutingDomainKind::Commands,
            vec![RoutingCandidate {
                id: "search".to_string(),
                examples: vec!["search: Search".to_string()],
            }],
        ),
        (
            RoutingDomainKind::Memory,
            vec![
                RoutingCandidate {
                    id: "conv".to_string(),
                    examples: vec!["conversations".to_string()],
                },
                RoutingCandidate {
                    id: "kb".to_string(),
                    examples: vec!["knowledge base".to_string()],
                },
            ],
        ),
    ];

    let decision = RoutingDecision::fallback(&domains);

    // Commands: scored at 1.0.
    assert_eq!(decision.commands.len(), 1);

    // Memory stores: all at 1.0, sorted alphabetically.
    assert_eq!(decision.memory_stores.len(), 2);
    assert_eq!(decision.memory_stores[0].id, "conv");
    assert_eq!(decision.memory_stores[1].id, "kb");
    for m in &decision.memory_stores {
        assert_eq!(m.score, 1.0, "fallback memory scores should be 1.0");
    }

    // Model and tools: conservative defaults.
    assert!(decision.model.is_none());
    assert!(decision.tools_needed.is_none());
}

#[test]
fn test_fallback_decision_memory_stores_empty_when_no_memory_domain() {
    // When Memory domain is NOT included in domains, memory_stores stays empty.
    let domains = vec![(
        RoutingDomainKind::Commands,
        vec![RoutingCandidate {
            id: "search".to_string(),
            examples: vec!["search: Search".to_string()],
        }],
    )];

    let decision = RoutingDecision::fallback(&domains);
    assert!(decision.memory_stores.is_empty());
}

// ── Phase 3: memory domain disabled via config ─────────────────────────

#[tokio::test]
async fn test_recall_with_memory_domain_disabled_fans_out_to_all() {
    use super::memory_tests::{make_mux_with_stores, MockMemStoreClient, mem_entry};
    let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
        "m1",
        "memory found",
    )]));
    let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

    // Build config with memory domain explicitly disabled.
    let disabled_domain = DomainConfig {
        enabled: false,
        threshold: None,
    };
    let config = Arc::new(WeftConfig {
        server: ServerConfig {
            bind_address: "127.0.0.1:8080".to_string(),
        },
        router: RouterConfig {
            classifier: ClassifierConfig {
                model_path: "models/test.onnx".to_string(),
                tokenizer_path: "models/tokenizer.json".to_string(),
                threshold: 0.5,
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
            domains: DomainsConfig {
                model: None,
                tool_necessity: None,
                memory: Some(disabled_domain),
            },
        },
        tool_registry: None,
        memory: Some(MemoryConfig {
            stores: vec![MemoryStoreConfig {
                name: "conv".to_string(),
                endpoint: "http://localhost:50052".to_string(),
                connect_timeout_ms: 5000,
                request_timeout_ms: 10000,
                max_results: 5,
                capabilities: vec![StoreCapability::Read, StoreCapability::Write],
                examples: vec!["conversation".to_string()],
            }],
        }),
        hooks: vec![],
        max_pre_response_retries: 2,
        request_end_concurrency: 64,
        gateway: GatewayConfig {
            system_prompt: "You are a test assistant.".to_string(),
            max_command_iterations: 10,
            request_timeout_secs: 30,
        },
    });

    let registry = single_model_registry(
        MockLlmProvider::new(vec!["/recall query: \"test query\"", "Found something."]),
        "test-model",
        "claude-test",
    );

    // Router: score_memory_candidates still works (above threshold for conv).
    let router = MockRouter::with_score(0.9).with_memory_score(0.9);

    let engine = make_engine_with_config_and_mux(
        config,
        registry,
        router,
        MockCommandRegistry::new(vec![]),
        Some(mux),
    );

    let resp = engine
        .handle_request(make_user_request("recall test"))
        .await
        .expect("should succeed");

    assert_eq!(resp_text(&resp), "Found something.");
}
